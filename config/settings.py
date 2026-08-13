
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=['settings.toml', 'paths.toml', '.secrets.toml'],
    # environments = True,
    # default_env = 'default',
    env_switcher  = "REMADE_ENV",
    envvar_prefix ="REMADE",
    load_dotenv   = True,
    env_nested_delimiter = "__",
    lowercase_read = True
)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
