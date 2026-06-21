# Geometry-aware contact cost (design notes)

Notes for a future iteration on the contact-avoidance term in
`peg_insertion_twist.py` (`reward()`). Not implemented yet — parked here.

## Current term

Isotropic Gaussian bump that repels the peg **tip centerline point** from a
contact hotspot `cp_xy` (judo-world xy, supplied via
`system_metadata["contact_pt"]`):

```
d_xy  = ‖tip_xy − cp_xy‖
cost += w_contact · exp(−½ (d_xy / r)²)        # r = contact_radius
```

Limitations:
- Ignores the peg's **finite cross-section** (contact is at the surface, not the centerline).
- Ignores the peg's **orientation** (rectangle/arch footprint is not radially symmetric).
- Ignores the **direction of binding** (repels radially in every lateral direction, so it
  fights insertion even along free directions).

## Proposed upgrades (cheap → rich)

### 1. Add the peg half-width (clearance offset)
Subtract an effective peg radius so the keep-out scales with the peg surface:

```
d_eff = ‖tip_xy − cp_xy‖ − R_peg                 # R_peg = peg cross-section half-width
cost += w_contact · exp(−½ (max(d_eff, 0) / r)²) # or softplus(−d_eff/r) for a one-sided wall
```

`R_peg` can be read once at build time from the peg collision-geom bbox.

### 2. Anisotropic (Mahalanobis) — encodes rectangular footprint + yaw
Use the peg's local half-extents `(a, b)` rotated by the peg yaw `θ` (from `tip_quat`):

```
δ   = tip_xy − cp_xy
δ'  = R(−θ) · δ                                  # bearing in peg-local frame
cost += w_contact · exp(−½ [ (δ'ₓ/(a+r))² + (δ'_y/(b+r))² ])
```

Ellipse-shaped keep-out matching the cross-section: wider where the peg is wide,
tight where it's thin. `(a, b)` come from the collision bbox (× `peg_coll_scale`).

### 3. One-sided along the contact normal (if available)
If `/contact_residuals` carries a normal `n` (or it's estimated from the cloud
gradient), penalize only motion *into* the contact, leaving tangential sliding free:

```
cost += w_contact · relu( n · (cp_xy − tip_xy) )
```

## Recommendation

Combine **1 + 2**: anisotropic Gaussian with `(a+r, b+r)` axes.
- ~2-line change in `reward()`, plus cache `self.peg_half_extents` in
  `_process_spec`/`__init__` and read peg yaw from `tip_quat`.
- Keeps the bounded-bump property (no gradient far away).
- Directly conditions on peg shape + orientation.

Add **3** only if a usable contact normal is available.
