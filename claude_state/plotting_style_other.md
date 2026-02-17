# Preferred Plotting Style

## Grid & Theme
- `sns.set_style("whitegrid")` + `plt.style.use("ggplot")`

## Font Sizes
- Title: 21pt bold
- Y-axis label: 20pt
- X-tick labels: 18pt, rotation=45, ha="right"
- Y-tick ticks: 17pt
- Suptitle: 26pt bold

## Bars + Intervals
- Light semi-transparent bar up to the mean: `ax.bar(..., color=color, alpha=0.35, edgecolor=color, linewidth=1.2)`
- Errorbar dot + CI on top: `ax.errorbar(..., fmt="o", markersize=10, capsize=6, capthick=2, linewidth=2.5)`
- Gray dashed zero line: `ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)`

## Colors
- Def. 1 / Unstrat.: `#4C72B0` (blue)
- Def. 2 / Strat.: `#C44E52` (red)

## Display Name Abbreviations
| Raw | Display |
|-----|---------|
| KIMIK2 | KIMI-K2 |
| Q80 | Qwen3-Next-80B |
| sequential | Sequential |
| zero_shot | Zero-Shot |
| specific | Specific |
| unspecific | Unspecific |
| original | Orig. |
| conflicting | Conf. |
| frustration_str | Frust. Monit. |
| frustration_llm | Frust. LLM Monit. |
| apologetic_str | Apol. Monit. |
| confident_str | Conf. Monit. |
| Def1 | Def. 1 |
| Def2 | Def. 2 |
| sus_score | Sus. Score |
| sus | Sus. |
| ti | Test Unintegrity |

## Config Label Format
`{Actor} / {Monitor} (Scaffold: {Scaffold}, Prompt: {Prompt})`
e.g. `KIMI-K2 / Qwen3-Next-80B (Scaffold: Sequential, Prompt: Specific)`

## Layout
- Use `matplotlib.gridspec.GridSpec` for multi-row layouts
- Shared y-axis within rows (dynamic min 0.5 upper bound)
- `tight_layout` with `bbox_inches="tight"` for saving
