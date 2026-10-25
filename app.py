"""
Particula que aprende a esquivar obstaculos moviles con DQN (PyTorch).

Uso:
    python particula.py train                # entrena y guarda particula.pt
    python particula.py train --episodes 3000
    python particula.py demo                 # muestra la animacion con el modelo entrenado
    python particula.py train --device cuda  # opcional: usar la Quadro P620
"""
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Entorno -----------------------------
N_RAYS = 16        # sensores tipo "lidar" alrededor de la particula
RAY_LEN = 0.4      # alcance de los sensores
N_OBS = 6          # numero de obstaculos
OBS_R = 0.05       # radio de obstaculos
PART_R = 0.02      # radio de la particula
ACC = 0.004        # aceleracion por accion
MAX_V = 0.03       # velocidad maxima de la particula
MAX_STEPS = 600    # pasos maximos por episodio
ACTIONS = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.float64)
OBS_DIM = N_RAYS * 2 + 4  # rayos actuales + rayos previos + velocidad + posicion


class Env:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
        self.dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    def reset(self):
        self.p = np.array([0.5, 0.5])
        self.v = np.zeros(2)
        pos = []
        while len(pos) < N_OBS:
            c = self.rng.uniform(OBS_R, 1 - OBS_R, 2)
            if np.linalg.norm(c - self.p) > 0.25:
                pos.append(c)
        self.o_pos = np.array(pos)
        th = self.rng.uniform(0, 2 * np.pi, N_OBS)
        sp = self.rng.uniform(0.006, 0.016, N_OBS)
        self.o_vel = np.stack([np.cos(th), np.sin(th)], axis=1) * sp[:, None]
        self.t = 0
        self.prev = self._rays()
        return self._obs()

    def _rays(self):
        f = self.p - self.o_pos                      # (N, 2)
        b = self.dirs @ f.T                          # (R, N)
        cc = (f * f).sum(1) - OBS_R ** 2             # (N,)
        disc = b * b - cc
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        t = np.where((disc >= 0) & (t > 0), t, np.inf)
        t_obs = t.min(axis=1)
        dx, dy = self.dirs[:, 0], self.dirs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (1 - self.p[0]) / dx, (0 - self.p[0]) / dx)
            ty = np.where(dy > 0, (1 - self.p[1]) / dy, (0 - self.p[1]) / dy)
        t_wall = np.minimum(tx, ty)
        return np.minimum(np.minimum(t_obs, t_wall), RAY_LEN) / RAY_LEN

    def _obs(self):
        cur = self._rays()
        obs = np.concatenate([cur, self.prev, self.v / MAX_V, (self.p - 0.5) * 2])
        self.prev = cur
        return obs.astype(np.float32)

    def step(self, a):
        self.t += 1
        self.v = (self.v + ACTIONS[a] * ACC) * 0.98
        n = np.linalg.norm(self.v)
        if n > MAX_V:
            self.v *= MAX_V / n
        self.p += self.v

        reward = 0.0
        for i in range(2):  # paredes: se frena, pequena penalizacion
            if self.p[i] < PART_R:
                self.p[i], self.v[i] = PART_R, 0.0
                reward -= 0.05
            elif self.p[i] > 1 - PART_R:
                self.p[i], self.v[i] = 1 - PART_R, 0.0
                reward -= 0.05

        self.o_pos += self.o_vel  # obstaculos rebotan en las paredes
        for i in range(2):
            lo = self.o_pos[:, i] < OBS_R
            hi = self.o_pos[:, i] > 1 - OBS_R
            self.o_vel[lo | hi, i] *= -1
            self.o_pos[:, i] = np.clip(self.o_pos[:, i], OBS_R, 1 - OBS_R)

        gap = np.linalg.norm(self.o_pos - self.p, axis=1) - OBS_R - PART_R
        if gap.min() <= 0:
            return self._obs(), reward - 1.0, True, False
        reward += 0.02 - 0.02 * max(0.0, 1 - gap.min() / 0.15)  # sobrevivir y mantener distancia
        truncated = self.t >= MAX_STEPS
        return self._obs(), reward, False, truncated


# ----------------------------- DQN -----------------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap, device):
        self.s = np.zeros((cap, OBS_DIM), np.float32)
        self.s2 = np.zeros((cap, OBS_DIM), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.i, self.n, self.dev = cap, 0, 0, device

    def add(self, s, a, r, s2, d):
        k = self.i
        self.s[k], self.a[k], self.r[k], self.s2[k], self.d[k] = s, a, r, s2, d
        self.i = (k + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        t = lambda x: torch.from_numpy(x[idx]).to(self.dev)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


def train(args):
    torch.set_num_threads(4)  # 4 nucleos fisicos
    dev = torch.device(args.device)
    env = Env()
    q, qt = QNet().to(dev), QNet().to(dev)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = Replay(200_000, dev)

    gamma, bs, tau = 0.97, 128, 0.005
    eps_start, eps_end = 1.0, 0.05
    eps_steps = int(args.episodes * 0.5 * 350)  # decae durante ~la mitad del entrenamiento
    step, hist, best = 0, deque(maxlen=50), -1

    for ep in range(1, args.episodes + 1):
        s, done, trunc, length = env.reset(), False, False, 0
        while not (done or trunc):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_steps)
            if random.random() < eps:
                a = random.randrange(len(ACTIONS))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s).to(dev)).argmax())
            s2, r, done, trunc = env.step(a)
            buf.add(s, a, r, s2, float(done))  # truncado NO es terminal
            s, step, length = s2, step + 1, length + 1

            if buf.n > 2000 and step % 2 == 0:
                bs_, ba, br, bs2, bd = buf.sample(bs)
                with torch.no_grad():  # Double DQN
                    a2 = q(bs2).argmax(1, keepdim=True)
                    y = br + gamma * (1 - bd) * qt(bs2).gather(1, a2).squeeze(1)
                pred = q(bs_).gather(1, ba[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    for pt, p in zip(qt.parameters(), q.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)

        hist.append(length)
        if ep % 50 == 0:
            m = np.mean(hist)
            print(f"ep {ep:5d} | supervivencia media {m:6.1f}/{MAX_STEPS} | eps {eps:.2f}")
            if m > best:
                best = m
                torch.save(q.state_dict(), "particula.pt")
    print(f"Listo. Mejor media: {best:.1f}. Modelo guardado en particula.pt")


# ----------------------------- Demo -----------------------------
def demo(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    q = QNet()
    q.load_state_dict(torch.load("particula.pt", map_location="cpu"))
    q.eval()
    env = Env()
    state = {"s": env.reset()}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
    part = Circle(env.p, PART_R, color="tab:blue")
    obs = [Circle(c, OBS_R, color="tab:red") for c in env.o_pos]
    ax.add_patch(part)
    for o in obs:
        ax.add_patch(o)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes)

    def update(_):
        with torch.no_grad():
            a = int(q(torch.from_numpy(state["s"])).argmax())
        s, r, done, trunc = env.step(a)
        state["s"] = env.reset() if (done or trunc) else s
        part.center = env.p
        for o, c in zip(obs, env.o_pos):
            o.center = c
        txt.set_text(f"paso {env.t}")
        return [part, txt, *obs]

    ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "demo"])
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    train(a) if a.mode == "train" else demo(a)