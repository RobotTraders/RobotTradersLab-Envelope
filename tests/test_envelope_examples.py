from importlib.resources import as_file, files

from robottraderslab import BotConfig


def test_backtest_example_config_loads():
    config_path = (
        files("robottraderslab_envelope") / "examples" / "envelope-bot-example.toml"
    )

    with as_file(config_path) as path:
        bot_config = BotConfig.from_file(str(path))

    assert bot_config.strategy.strategy_class == "envelope"
