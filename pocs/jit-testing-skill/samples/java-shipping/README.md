# java-shipping sample

Parent (in `.parent/`): free shipping when `total >= $50`.

Current (buggy) version: adds free shipping for pickup orders, but the threshold for delivery orders was accidentally lowered from `5000` cents to `500` cents. `/jit` should catch a $49.99 delivery order behavior change.

## Run

```
/jit
```

That's it. The skill auto-detects snapshot mode because `.parent/` exists. No git setup required.

If you want to exercise git mode instead, run `./setup-git.sh` first to materialize a real parent → diff commit history.
