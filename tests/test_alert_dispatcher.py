import asyncio
import importlib.util
import sys
from pathlib import Path


def load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "افضل واحد للان (1) (٣).py"
    spec = importlib.util.spec_from_file_location("smart_money_algo", module_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    sys.modules.setdefault(spec.name, module)
    loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_alert_dispatcher_emits_idm_before_delayed_events():
    module = load_module()
    AlertDispatcher = module.AlertDispatcher
    ImmediateEventCache = module.ImmediateEventCache
    _ = ImmediateEventCache  # noqa: F841 - ensure symbol reachable for documentation

    sink: list[str] = []

    async def runner() -> None:
        lock = asyncio.Lock()
        dispatcher = AlertDispatcher(lock, sink=sink.append)
        await dispatcher.start()

        async def fast_symbol() -> None:
            await dispatcher.publish(
                "BTC/USDT",
                {
                    "IDM_OB": {
                        "time": 1,
                        "display": "IDM OB ready",
                        "status": "new",
                        "status_display": "new",
                    }
                },
            )

        async def slow_symbol() -> None:
            await asyncio.sleep(0.1)
            await dispatcher.publish(
                "ETH/USDT",
                {
                    "EXT_OB": {
                        "time": 2,
                        "display": "EXT OB touch",
                        "status": "retest",
                        "status_display": "retest",
                    }
                },
            )

        await asyncio.gather(fast_symbol(), slow_symbol())
        await dispatcher.close()

    asyncio.run(runner())

    assert len(sink) == 2
    first, second = sink
    assert "IDM_OB" in first
    assert "EXT_OB" in second
    assert sink[0].split("—")[0].strip().startswith("["), "timestamp prefix missing"
