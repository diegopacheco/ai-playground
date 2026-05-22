# python-pricing sample

Parent (in `.parent/`): free shipping when `total >= 5000`.
Current: adds pickup branch, but the threshold for delivery was lowered to `500`. `/jit` should catch the regression.

## Run

```
/jit
```

Snapshot mode is auto-selected because `.parent/` is present. No git setup required.
