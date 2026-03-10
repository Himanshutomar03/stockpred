import subprocess, os

GIT = r"C:\Program Files\Git\bin\git.exe"
REPO = r"e:\stockpred"
ENV = {**os.environ, "GIT_EDITOR": "true", "GIT_TERMINAL_PROMPT": "0"}

def run(args):
    cmd = [GIT] + args
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=ENV, timeout=30)
    out = (r.stdout + r.stderr).strip()
    print(f"$ git {' '.join(args)}\n{out}\nRC={r.returncode}\n---")
    return r

run(["config", "user.email", "user@stockedge.local"])
run(["config", "user.name", "StockEdgeUser"])
run(["add", "app.py"])
rc = run(["commit", "-m",
          "feat: premium UI redesign v2.0 - glassmorphism animations ticker marquee glow cards"])
rv = run(["remote", "-v"])
if not rv.stdout.strip():
    print("NO_REMOTE: no remote configured - cannot push.")
else:
    run(["push"])
