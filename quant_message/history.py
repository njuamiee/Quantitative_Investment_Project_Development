"""Historical clock and dated entitlement events, for both execution engines."""
import json
from pathlib import Path
from .paper import PaperBroker


class ReplayActions:
    def configure_actions(self, owner):
        path=getattr(owner,'corporate_action_path','')
        actions=json.loads(Path(path).read_text()) if path else []
        self.replay_actions=sorted([a for a in actions if a['symbol'] in self.instruments],key=lambda a:a['date'])
        if any(a['ratio']!=1 for a in self.replay_actions):
            raise RuntimeError('native share-split position reconciliation is not implemented')

    def advance(self, now):
        super().advance(now)
        while self.replay_actions and self.replay_actions[0]['date']<=self.day.isoformat():
            a=self.replay_actions.pop(0)
            self.corporate_action(a['date']+':'+a['symbol'],a['symbol'],self.now,
                                  a['dividend'],a['ratio'],credit_cash=False)


class PaperReplayBroker(ReplayActions,PaperBroker):
    def __init__(self,owner,instruments,capital,costs):
        from vnpy_deltaone.backtesting import BacktestingEngine
        if not isinstance(owner.engine,BacktestingEngine):
            raise TypeError('historical clock requires BacktestingEngine')
        super().__init__(instruments,capital,costs)
        self.configure_actions(owner)
