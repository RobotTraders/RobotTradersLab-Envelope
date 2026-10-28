from importlib.resources import as_file, files
from pathlib import Path

from robottraderslab import BotConfig
from robottraderslab.scaffold import main as scaffold


def test_backtest_example_config_loads():
    config_path = files("robottraderslab_envelope") / "examples" / "bot.toml"

    with as_file(config_path) as path:
        bot_config = BotConfig.from_file(str(path))

    assert bot_config.strategy.strategy_class == "envelope"


def test_backtest_example_profiles_build(make_strategy):
    config_path = files("robottraderslab_envelope") / "examples" / "bot.toml"

    with as_file(config_path) as path:
        bot_config = BotConfig.from_file(str(path))

    strategy = make_strategy(bot_config.strategy.profiles)

    assert len(strategy.profiles) == len(bot_config.strategy.profiles)


def test_scaffold_creates_a_bot_folder_with_its_notebook(tmp_path):
    scaffold("envelope", Path("first-bot"), tmp_path)
    folder = tmp_path / "first-bot"

    assert {path.name for path in folder.iterdir()} >= {
        "bot.toml",
        "run_backtest.ipynb",
    }
