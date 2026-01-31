<<<<<<< HEAD
# ChristmasTreePackingGame
This is a tree packing sandbox studio/software. 
=======
# Tree Packing Studio v10

## Overview

Tree Packing Studio v10 is the refined, high-precision follow-up to the Kaggle **Santa 2025: Tree Packing Challenge**. It lets you:

- Load any previously recorded configuration (`N=001`..`200`) from `comparison/` with a single press.  
- Fine-tune positions and rotations with sub-millimeter keyboard nudges, rotate increments, and a "pan" magnifier mode.  
- Run the full optimizer or the new Auto-O multi-pass helper with controlled pass counts.  
- Record, log, and snapshot layouts with canonical tracking, while comparison tools let you score and compare submission files offline.

It was built to be **100% deterministic** in bookkeeping, **100% precise** in editing, and easy for collaborators to drop into an open-source repo.

## Directory layout

```
TreePackingStudioV10/
├── tree_sandbox_v10.py     # Main app (pygame + Shapely)
├── game_v10/              # Outputs (dynamic; created on run)
├── comparison/            # Put your recorded `submitted1.csv`, `raw.csv`, etc.
├── tools/
│   ├── compare_n_s.py
│   ├── compare_submissions.py
│   ├── whatsmyscore.py
│   └── n1_opt.py
└── README.md
```

## Run the studio

Design your workflow:

1. Place your `submitted1.csv`, `raw.csv`, or any Kaggle-style CSV inside `comparison/`.  
2. Run `python tree_sandbox_v10.py`. Outputs are written under `game_v10/`, including `layout_kaggle.csv`, `canonical/canonical.csv`, snapshots, and automation logs.
3. Use the UI buttons (+ hotkeys) to:
   - `G` Generate / Auto-place  
   - `O` Optimize (restores the best layout when finished)  
   - `T` Auto-O (runs the chosen optimize pass count)  
   - `M` Enter pan mode for the magnifier  
   - `Alt+Wheels` Zoom to cursor, `Ctrl+Wheel` to rotate, and `Alt+Arrows` for precise nudges  
   - `K` Load a configuration: type the CSV file name (just the name, because it already looks in `comparison/`), click `Set`, then `Load N (K)`.
4. Record (`L`) to log snapshots + canonical updates, keeping `high score` in sync.

## Tools

- `tools/compare_n_s.py`: Compare `S` per `N` between any two CSVs.  
- `tools/compare_submissions.py`: Full summary with `s`, `s²`, and which layout wins.  
- `tools/whatsmyscore.py`: Computes the Kaggle metric (`Σ S² / N`).  
- `tools/n1_opt.py`: Explore the optimal rotation for `N=1` (leveled rotation search).

All tools reference this repository's `comparison/` folder by default.

## Precision & Optimizer

The optimizer stores its start layout and snaps back to the best configuration when a pass finishes. The XY/rotation physics are precise (sub-degree rotation, 0.0001 units nudges) and allow you to align tree edges with confidence.  

### Keyboard shortcuts reminder

- `Ctrl+Mouse Wheel`: Zoom toward cursor  
- `Mouse Wheel`: Rotate hovered tree  
- `Alt+Arrow`: Micro-nudge selected tree  
- `[ / ]`: Rotate selected tree  
- `M`: Toggle pan mode  
- `T`: Run auto multi-pass optimize  
- `F9`: Run full automation sweep  
- `K`: Load `N` from comparison CSV  
- `L`: Record  

## Acknowledgments

This studio is inspired by the Santa 2025 Kaggle challenge. While the final submission window closed, we package our tools and data so others can reproduce or extend the effort. This code is released under **CC BY 4.0**; feel free to fork, improve, and share.
