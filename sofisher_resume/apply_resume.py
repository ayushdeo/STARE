"""
Add resume-from-checkpoint to a local SOFisher clone (Heathcliff-Ng/SOFisher).

Faithful modification: adds PERSISTENCE ONLY (a full checkpoint: q_eval, q_target, optimizer, epsilon,
total_steps, best_eval_rew, reward history) and a --resume flag. It changes no hyperparameter, reward, or
training logic, so runs remain faithful to the authors' setup. Idempotent; backs up originals to *.orig.

Why you need it: SOFisher's default max_steps = 5,000,000, and its save() persists only network weights
(no optimizer / step counter / epsilon), with no resume path. So a restart loses the optimizer state,
the exploration schedule, and the step count. This patch lets you stop and continue.

Usage (on the 4070, from anywhere):
    python sofisher_resume/apply_resume.py /path/to/SOFisher
Then in the SOFisher clone:
    python main_cortex.py                                             # writes seed<SEED>_resume.pth periodically
    python main_cortex.py --resume model/DQN/seed_222/seed222_resume.pth   # continue
    # find the exact file:  find . -name 'seed*_resume.pth'
"""
import sys, os, shutil

ARGS_INSERT_BEFORE = "    args, unknown = parser.parse_known_args()"
ARGS_NEW = (
    "    parser.add_argument('--resume', type=str, default='', help='path to a full resume checkpoint (.pth)')\n"
    "    parser.add_argument('--warmstart', type=str, default='', help=\"name of existing weights to warm-start from (e.g. 'final' or 'best')\")\n"
    "    parser.add_argument('--ckpt_every', type=int, default=50000, metavar='N',\n"
    "                        help='steps between full resume checkpoints')\n"
)

WHILE_ANCHOR = "        while self.total_steps < self.args.max_steps:"
RESUME_LOAD = (
    "        if getattr(self.args, 'resume', '') and os.path.exists(self.args.resume):\n"
    "            self.load_resume(self.args.resume)\n"
    "        elif getattr(self.args, 'warmstart', ''):\n"
    "            self.agent.load(self.seed, self.args.warmstart, self.model_seed_folder)\n"
    "            self.agent.epsilon = self.agent.eps_min\n"
    "            print('[warmstart] loaded {} weights; epsilon set to floor {}'.format(self.args.warmstart, self.agent.epsilon))\n"
)

SAVE_ANCHOR = "            np.save('./{}/train_episode_reward.npy'.format(self.data_train_folder), self.train_rewards)"
SAVE_INSERT = (
    "\n            if getattr(self.args, 'ckpt_every', 0) and (self.total_steps % self.args.ckpt_every) < self.args.episode_size:\n"
    "                self.save_resume()"
)

METHODS_ANCHOR = "if __name__ == '__main__':"
METHODS = (
    "    def _resume_path(self):\n"
    "        return os.path.join(self.model_seed_folder, 'seed{}_resume.pth'.format(self.seed))\n"
    "\n"
    "    def save_resume(self):\n"
    "        torch.save({\n"
    "            'q_eval': self.agent.q_eval.state_dict(),\n"
    "            'q_target': self.agent.q_target.state_dict(),\n"
    "            'optimizer': self.agent.q_eval.optimizer.state_dict(),\n"
    "            'epsilon': self.agent.epsilon,\n"
    "            'total_steps': self.total_steps,\n"
    "            'best_eval_rew': self.best_eval_rew,\n"
    "            'evaluate_rewards': self.evaluate_rewards,\n"
    "            'train_rewards': self.train_rewards,\n"
    "        }, self._resume_path(), _use_new_zipfile_serialization=False)\n"
    "        print('[resume] checkpoint saved at {} steps -> {}'.format(self.total_steps, self._resume_path()))\n"
    "\n"
    "    def load_resume(self, path):\n"
    "        dev = next(self.agent.q_eval.parameters()).device\n"
    "        ck = torch.load(path, map_location=dev)\n"
    "        self.agent.q_eval.load_state_dict(ck['q_eval'])\n"
    "        self.agent.q_target.load_state_dict(ck['q_target'])\n"
    "        self.agent.q_eval.optimizer.load_state_dict(ck['optimizer'])\n"
    "        self.agent.epsilon = ck['epsilon']\n"
    "        self.total_steps = ck['total_steps']\n"
    "        self.best_eval_rew = ck['best_eval_rew']\n"
    "        self.evaluate_rewards = ck['evaluate_rewards']\n"
    "        self.train_rewards = ck.get('train_rewards', [])\n"
    "        print('[resume] restored at {} steps, eps={:.4f}'.format(self.total_steps, self.agent.epsilon))\n"
    "\n\n"
)


def patch_file(path, edits, sentinel):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if sentinel in src:
        print(f"[skip] {path} already patched")
        return False
    for anchor, newtext, mode in edits:
        if anchor not in src:
            raise SystemExit(f"[error] anchor not found in {path}:\n  {anchor!r}\n"
                             f"Repo may differ from expected. No changes written.")
        if mode == "before":
            src = src.replace(anchor, newtext + anchor, 1)
        elif mode == "after":
            src = src.replace(anchor, anchor + newtext, 1)
    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[patched] {path}  (backup: {os.path.basename(path)}.orig)")
    return True


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    args_py = os.path.join(root, "src", "args.py")
    main_py = os.path.join(root, "main_cortex.py")
    for p in (args_py, main_py):
        if not os.path.exists(p):
            raise SystemExit(f"[error] not found: {p}\nPass the SOFisher repo root as argument.")

    patch_file(args_py, [(ARGS_INSERT_BEFORE, ARGS_NEW, "before")], sentinel="--resume")
    patch_file(main_py, [
        (METHODS_ANCHOR, METHODS, "before"),     # add save_resume/load_resume as Runner methods
        (WHILE_ANCHOR, RESUME_LOAD, "before"),   # load resume before the training loop
        (SAVE_ANCHOR, SAVE_INSERT, "after"),     # periodic full checkpoint
    ], sentinel="def save_resume")

    print("\nDone. Resume works like:")
    print("  python main_cortex.py                 # trains, writes seed<SEED>_resume.pth every --ckpt_every steps")
    print("  find . -name 'seed*_resume.pth'       # locate the checkpoint (SEED is hardcoded 222 in main_cortex)")
    print("  python main_cortex.py --resume <that_path>   # continue")
    print("Note: the replay buffer is not persisted; it refills on resume while epsilon is at its floor "
          "(minor). Other apply_* mains (aging/disease/CRC) share the Runner pattern but are not auto-patched.")


if __name__ == "__main__":
    main()
