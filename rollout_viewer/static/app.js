const $ = (q, el=document) => el.querySelector(q);
const $$ = (q, el=document) => Array.from(el.querySelectorAll(q));

const state = {
  currentPath: null,
  page: 1,
  pageSize: 50,
  loading: false,
  hasMore: false,
  q: '',
  cursor: 0,
  filterKey: '',
};

async function api(path, params={}){
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k,v]) => url.searchParams.set(k, v));
  const res = await fetch(url.toString());
  if(!res.ok) throw new Error(await res.text());
  return res.json();
}

function humanSize(bytes){
  if(bytes == null) return '';
  const units = ['B','KB','MB','GB'];
  let i=0, v=bytes;
  while(v>1024 && i<units.length-1){ v/=1024; i++; }
  return `${v.toFixed(1)} ${units[i]}`;
}

function renderTree(container, nodes){
  container.innerHTML = '';
  nodes.entries.forEach(item => {
    const row = document.createElement('div');
    row.className = 'entry';
    row.dataset.path = item.path;
    row.dataset.type = item.type;
    row.innerHTML = `
      <div class="icon">${item.type==='dir' ? '📁' : '📄'}</div>
      <div class="name ${item.type==='file' ? 'file':''}">${item.name}</div>
      <div style="margin-left:auto; color:#9aa4b2; font-size:12px">${item.type==='file' ? humanSize(item.size) : ''}</div>
    `;
    row.addEventListener('click', async () => {
      if(item.type==='dir'){
        const data = await api('/api/tree', { path: item.path });
        renderTree(container, data);
        updateBreadcrumb(item.path);
        try { history.pushState({type:'dir', path:item.path}, '', `#dir=${encodeURIComponent(item.path)}`); } catch {}
      } else if(item.type==='file' && item.name.endsWith('.jsonl')){
        try { history.pushState({type:'file', path:item.path}, '', `#file=${encodeURIComponent(item.path)}`); } catch {}
        openJsonl(item.path);
      }
    });
    container.appendChild(row);
  });
}

function updateBreadcrumb(path){
  $('#breadcrumb').textContent = path || '';
}

function renderEntryCard(entry){
  const card = document.createElement('div');
  card.className = 'card';
  const obj = entry.obj;
  const raw = entry.raw;
  const kvs = [];
  if(obj && typeof obj === 'object'){
    for(const [k,v] of Object.entries(obj)){
      if(v == null) continue;
      if(typeof v === 'string' && v.length > 200){
        kvs.push({k, v, isText:true});
      } else if(typeof v === 'object'){
        kvs.push({k, v: JSON.stringify(v, null, 2), isText:true});
      } else {
        kvs.push({k, v: String(v), isText:false});
      }
    }
  } else {
    kvs.push({k:'raw', v: raw, isText:true});
  }

  const grid = document.createElement('div');
  grid.className = 'kv-grid';
  kvs.forEach(({k,v,isText}, idx) => {
    const row = document.createElement('div');
    row.className = 'row';
    const key = document.createElement('div');
    key.className = 'key';
    key.textContent = k;
    const value = document.createElement('div');
    value.className = 'value';
    if(isText){
      const wrap = document.createElement('div');
      wrap.className = 'text-wrap';
      const box = document.createElement('div');
      const tone = toneForKey(k);
      box.className = `text-box tone-${tone}`;
      box.textContent = v;
      const tb = document.createElement('div');
      tb.className = 'text-toolbar';
      const toggle = document.createElement('button');
      toggle.textContent = 'Expand';
      toggle.addEventListener('click', () => {
        box.classList.toggle('expanded');
        toggle.textContent = box.classList.contains('expanded') ? 'Collapse' : 'Expand';
      });
      tb.appendChild(toggle);
      wrap.appendChild(tb);
      wrap.appendChild(box);
      value.appendChild(wrap);
    } else {
      const span = document.createElement('span');
      const tone = toneForKey(k);
      span.className = `pill tone-${tone}`;
      span.textContent = v;
      value.appendChild(span);
    }
    row.appendChild(key);
    row.appendChild(value);
    grid.appendChild(row);
  });
  card.appendChild(grid);
  return card;
}

function toneForKey(key){
  // Simple deterministic hash to 0..11
  let h = 0;
  for(let i=0;i<key.length;i++){
    h = (h*31 + key.charCodeAt(i)) >>> 0;
  }
  return h % 12;
}

async function openJsonl(path){
  state.currentPath = path;
  state.page = 1;
  state.hasMore = false;
  state.q = $('#searchBox')?.value || '';
  state.cursor = 0;
  state.filterKey = $('#filterKey')?.value || '';
  $('#entries').innerHTML = '';
  $('#fileMeta').textContent = `${path}`;
  updateBreadcrumb(path);
  await loadKeys();
  await loadMore();
}

async function loadMore(){
  if(state.loading || !state.currentPath) return;
  state.loading = true;
  $('#loadMore').textContent = 'Loading...';
  $('#loadMore').hidden = false;
  let params = { path: state.currentPath };
  const val = (state.q || '').trim();
  const fkey = (state.filterKey || '').trim();
  if(fkey && val){
    params.filter_key = fkey;
    params.filter_value = val;
    params.cursor = state.cursor;
    params.page_size = state.pageSize;
  } else if(val){
    params.q = state.q;
    params.cursor = state.cursor;
    params.page_size = state.pageSize;
  } else {
    params.page = state.page;
    params.page_size = state.pageSize;
  }
  const data = await api('/api/jsonl', params);
  state.hasMore = data.has_more || data.hasMore;
  const entriesEl = $('#entries');
  data.entries.forEach(e => entriesEl.appendChild(renderEntryCard(e)));
  if((fkey && val) || val){
    state.cursor = data.next_cursor ?? state.cursor;
  } else {
    state.page += 1;
  }
  state.loading = false;
  $('#loadMore').textContent = state.hasMore ? 'Load more' : 'No more';
  if(!state.hasMore) $('#loadMore').disabled = true;
}

async function loadKeys(){
  if(!state.currentPath) return;
  try{
    const data = await api('/api/jsonl_keys', { path: state.currentPath, sample: 1000 });
    const sel = $('#filterKey');
    sel.innerHTML = '<option value="">Any</option>' + data.keys.map(k => `<option value="${k}">${k}</option>`).join('');
    if(state.filterKey){ sel.value = state.filterKey; }
  }catch(e){ /* ignore */ }
}

async function init(){
  const root = await api('/api/root');
  $('#rootPath').textContent = root.root;
  const treeData = await api('/api/tree');
  renderTree($('#tree'), treeData);

  $('#pageSize').addEventListener('change', (e) => {
    state.pageSize = Math.max(1, Math.min(500, Number(e.target.value) || 50));
    if(state.currentPath){
      openJsonl(state.currentPath);
    }
  });

  $('#loadMore').addEventListener('click', loadMore);

  // Infinite scroll
  $('#content').addEventListener('scroll', () => {
    const el = $('#content');
    if(el.scrollTop + el.clientHeight >= el.scrollHeight - 200){
      if(state.hasMore && !state.loading){ loadMore(); }
    }
  });

  // Back button
  $('#navBack').addEventListener('click', () => {
    history.back();
  });

  // Live search with debounce
  let t;
  $('#searchBox').addEventListener('input', (e) => {
    clearTimeout(t);
    t = setTimeout(() => {
      state.q = e.target.value;
      if(state.currentPath){
        state.cursor = 0;
        $('#entries').innerHTML = '';
        state.hasMore = false;
        loadMore();
      }
    }, 250);
  });

  // Filter key change
  $('#filterKey').addEventListener('change', (e) => {
    state.filterKey = e.target.value || '';
    if(state.currentPath){
      state.cursor = 0;
      $('#entries').innerHTML = '';
      state.hasMore = false;
      loadMore();
    }
  });

  // Restore from URL hash
  const hash = window.location.hash;
  if(hash.startsWith('#file=')){
    const p = decodeURIComponent(hash.slice(6));
    openJsonl(p);
  } else if(hash.startsWith('#dir=')){
    const p = decodeURIComponent(hash.slice(5));
    const data = await api('/api/tree', { path: p });
    renderTree($('#tree'), data);
    updateBreadcrumb(p);
  }

  window.addEventListener('popstate', async (e) => {
    const st = e.state;
    if(!st){ return; }
    if(st.type === 'file'){
      openJsonl(st.path);
    } else if(st.type === 'dir'){
      const data = await api('/api/tree', { path: st.path });
      renderTree($('#tree'), data);
      updateBreadcrumb(st.path);
    }
  });

  // Tree filter (client-side, current directory only)
  $('#treeFilter').addEventListener('input', (e) => {
    const q = (e.target.value || '').toLowerCase();
    $$('.tree .entry').forEach(el => {
      const name = el.querySelector('.name')?.textContent?.toLowerCase() || '';
      el.style.display = name.includes(q) ? '' : 'none';
    });
  });
}

window.addEventListener('DOMContentLoaded', init);


