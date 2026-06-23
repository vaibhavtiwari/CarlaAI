if __package__ in (None, ""):
    import _bootstrap
else:
    from . import _bootstrap

from learned_policies.rl.ppo.train import main


if __name__ == "__main__":
    main()
