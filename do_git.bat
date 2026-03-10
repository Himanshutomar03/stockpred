@echo off
cd /d e:\stockpred
git config user.email "user@stockedge.local" >> e:\stockpred\git_out.log 2>&1
git config user.name "StockEdge User" >> e:\stockpred\git_out.log 2>&1
git add app.py >> e:\stockpred\git_out.log 2>&1
git commit -m "feat: premium UI redesign v2.0 - glassmorphism animations ticker marquee glow cards" >> e:\stockpred\git_out.log 2>&1
echo DONE >> e:\stockpred\git_out.log 2>&1
