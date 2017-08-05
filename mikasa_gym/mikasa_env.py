import pandas as pd
import numpy as np
import gym
from gym import error, spaces, utils
from sklearn.preprocessing import StandardScaler

from mikasa import *

ACTION_LOOKUP = {
    0: 'nop',
    1: 'buy',
    2: 'sell'
}


class MikasaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_filename=None, look_back=1, fields=None, balance=1000.0):
        if fields is None:
            fields = [
                'open',
                'high',
                'low',
                'close'
            ]
        if not source_filename:
            raise NotImplemented
        self.source_filename = source_filename
        self.fields = fields
        self.look_back = look_back
        self.balance = balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(fields) + 1, ))

        self.scaler = StandardScaler()

    def _get_reward(self):
        return self.bt.get_profit()

    def _step(self, action):
        reward = self._take_action(action)
        ob = self._get_observation()
        episode_over = self.ds.is_end()
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        reward = 0.0
        if ACTION_LOOKUP[action] == 'buy' and not self.bt.position:
            self.bt.buy(self.ds[0].close, self.balance)
        if ACTION_LOOKUP[action] == 'sell' and self.bt.position:
            self.bt.sell(self.ds[0].close)
            reward = self.bt.trades[-1].get_profit()
        if not self.bt.ds.is_end():
            self.bt.go()
        return reward

    def _reset(self):
        df = pd.read_csv(self.source_filename).rename(columns={
            'Close': 'close',
            'Date time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })
        df[self.fields] = self.scaler.fit_transform(df[self.fields])
        self.balance = self.balance
        self.ds = DataSeries(df, index=self.look_back)
        self.bt = BT(self.ds, self.balance)
        return self._get_observation()

    def _get_observation(self):
        ob = []
        for n in range(self.look_back):
            for k in self.fields:
                ob.append(getattr(self.ds[n-(self.look_back-1)], k))
        ob.append(1.0 if self.bt.position is None else 0.0)
        return ob

    def _render(self, mode='human', close=False):
        pass
