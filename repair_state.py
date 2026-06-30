"""
repair_state.py — Reconstruit state.json et nav_history.csv à partir des données réelles.

Ce script:
1. Lit tous les fichiers signal JSON (ils ont entry_price et qty pour les entrées)
2. Pour les exits, lit le prix de clôture depuis les CSV de prix (ALT_USDT/)
3. Reconstruit correctement: cash, realized_pnl, et corrige nav_history

À exécuter sur le serveur de prod APRÈS avoir pushé les fixes de code.
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime

# ── Paths ──
def get_project_root():
    # Ce script est à la racine du projet
    return os.path.dirname(os.path.abspath(__file__))

def main():
    project_root = get_project_root()
    data_dir      = os.path.join(project_root, "data", "crypto")
    signals_dir   = os.path.join(data_dir, "signals")
    alt_usdt_dir  = os.path.join(data_dir, "ALT_USDT")
    state_path    = os.path.join(data_dir, "state.json")
    nav_path      = os.path.join(data_dir, "nav_history.csv")

    print("=" * 60)
    print("  REPAIR STATE — Reconstruction depuis les signaux")
    print("=" * 60)

    if not os.path.exists(signals_dir):
        print("❌ Dossier signals introuvable:", signals_dir)
        return

    # ── 1. Lire tous les fichiers signal en ordre chronologique ──
    signal_files = sorted(glob.glob(os.path.join(signals_dir, "*.json")))
    print(f"\n📂 {len(signal_files)} fichiers signal trouvés")

    # ── 2. Fonction pour lire le prix de clôture d'un symbole à une date ──
    price_cache = {}

    def get_close_price(symbol, date_str):
        """Lire le prix de clôture d'un CSV pour une date donnée."""
        if symbol not in price_cache:
            csv_path = os.path.join(alt_usdt_dir, f"{symbol}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, parse_dates=["date"])
                df = df.set_index("date")
                price_cache[symbol] = df
            else:
                price_cache[symbol] = None

        df = price_cache.get(symbol)
        if df is None:
            return None

        try:
            target_date = pd.Timestamp(date_str)
            if target_date in df.index:
                return float(df.loc[target_date, "close"])
            # Chercher la date la plus proche
            closest = df.index[df.index.get_indexer([target_date], method="nearest")[0]]
            return float(df.loc[closest, "close"])
        except Exception as e:
            print(f"  ⚠️ Impossible de lire le prix {symbol} @ {date_str}: {e}")
            return None

    # ── 3. Rejouer les trades chronologiquement ──
    INITIAL_CASH = 10000.0
    cash         = INITIAL_CASH
    realized_pnl = 0.0
    positions    = {}   # {symbol_side: {symbol, side, entry_price, qty, entry_date}}
    underperf_streaks = {}

    nav_repairs = []  # Pour corriger nav_history

    print("\n📈 Replay des trades:\n")

    for sf in signal_files:
        with open(sf) as f:
            sig = json.load(f)

        date = sig.get("date", os.path.basename(sf).replace(".json", ""))

        # ── Exits ──
        for ex in sig.get("exits", []):
            symbol = ex["symbol"]
            side   = ex["side"]
            key    = f"{symbol}_{side}"

            if key not in positions:
                print(f"  ⚠️ EXIT {symbol} ({side}) @ {date} — position introuvable, skip")
                continue

            pos = positions[key]
            qty = pos["qty"]
            entry_price = pos["entry_price"]

            # Prix de sortie: d'abord regarder si le signal l'a (nouveau code)
            exit_price = ex.get("exit_price")
            if not exit_price:
                # Fallback: lire le close price du CSV
                exit_price = get_close_price(symbol, date)
            if not exit_price:
                # Dernier recours: utiliser entry_price (PnL = 0)
                exit_price = entry_price
                print(f"  ⚠️ Impossible de trouver le prix de sortie pour {symbol} @ {date}, PnL = 0")

            # Calcul PnL
            if side == "long":
                trade_pnl = (exit_price - entry_price) * qty
                exit_value = exit_price * qty
            else:  # short
                trade_pnl = (entry_price - exit_price) * qty
                exit_value = entry_price * qty + trade_pnl

            cash         += exit_value
            realized_pnl += trade_pnl

            print(f"  🔴 EXIT {side.upper()} {symbol} @ {date}")
            print(f"     entry=${entry_price:.4f}  exit=${exit_price:.4f}  qty={qty:.2f}")
            print(f"     PnL=${trade_pnl:+.2f}  cash back=${exit_value:.2f}  cash total=${cash:.2f}")

            del positions[key]
            underperf_streaks.pop(key, None)

            # Enrichir le signal avec les prix pour les prochains runs
            if not ex.get("exit_price"):
                ex["exit_price"] = round(exit_price, 8)
                ex["entry_price"] = round(entry_price, 8)
                ex["qty"] = round(qty, 6)
                ex["trade_pnl"] = round(trade_pnl, 4)
                ex["exit_value"] = round(exit_value, 4)

        # ── Entries ──
        for en in sig.get("entries", []):
            symbol      = en["symbol"]
            side        = en["side"]
            key         = f"{symbol}_{side}"
            entry_price = en.get("entry_price")
            qty         = en.get("qty", 0)
            size_cash   = en.get("size_cash", cash)  # montant investi

            if not entry_price:
                entry_price = get_close_price(symbol, date)
            if not entry_price:
                print(f"  ⚠️ Impossible de trouver le prix d'entrée pour {symbol} @ {date}, skip")
                continue

            if qty == 0 and entry_price > 0:
                qty = size_cash / entry_price

            positions[key] = {
                "symbol":      symbol,
                "side":        side,
                "entry_price": entry_price,
                "qty":         qty,
                "entry_date":  date,
                "peak":        entry_price if side == "long" else None,
                "trough":      entry_price if side == "short" else None,
            }
            cash -= size_cash

            print(f"  🟢 ENTRY {side.upper()} {symbol} @ {date}")
            print(f"     entry=${entry_price:.4f}  qty={qty:.2f}  invested=${size_cash:.2f}  cash=${cash:.2f}")

        # NAV pour ce jour
        open_positions_list = list(positions.values())
        nav_this_day = cash
        for pos in open_positions_list:
            sym   = pos["symbol"]
            px    = get_close_price(sym, date) or pos["entry_price"]
            q     = pos["qty"]
            ep    = pos["entry_price"]
            if pos["side"] == "long":
                nav_this_day += px * q
            else:
                nav_this_day += ep * q + (ep - px) * q
        nav_repairs.append({"date": date, "nav_corrected": round(nav_this_day, 2)})

        # Réécrire le fichier signal avec les prix si on les a ajoutés
        with open(sf, "w") as f:
            json.dump(sig, f, indent=2, default=str)

    # ── 4. Reconstruire state.json ──
    new_state = {
        "positions": [
            {k: v for k, v in pos.items() if v is not None}
            for pos in positions.values()
        ],
        "cash":             round(cash, 4),
        "initial_cash":     INITIAL_CASH,
        "realized_pnl":     round(realized_pnl, 4),
        "underperf_streaks": underperf_streaks,
    }

    print("\n" + "=" * 60)
    print("  ÉTAT RECONSTRUIT")
    print("=" * 60)
    print(f"  Cash:          ${cash:,.2f}")
    print(f"  Realized PnL:  ${realized_pnl:+,.2f}  ({realized_pnl/INITIAL_CASH:+.2%})")
    print(f"  Open positions: {len(positions)}")
    for key, pos in positions.items():
        print(f"    {pos['side'].upper()} {pos['symbol']} entry=${pos['entry_price']:.4f} qty={pos['qty']:.2f}")

    # Backup
    import shutil
    if os.path.exists(state_path):
        shutil.copy(state_path, state_path + ".bak_repair")
        print(f"\n📦 Backup state.json → state.json.bak_repair")

    with open(state_path, "w") as f:
        json.dump(new_state, f, indent=2, default=str)
    print(f"✅ state.json réparé et sauvegardé")

    # ── 5. Corriger nav_history.csv ──
    if os.path.exists(nav_path):
        nav_df = pd.read_csv(nav_path, parse_dates=["date"])

        repair_map = {r["date"]: r["nav_corrected"] for r in nav_repairs}

        def fix_nav(row):
            date_str = str(row["date"])[:10]
            if date_str in repair_map:
                return repair_map[date_str]
            return row["nav"]

        nav_df["nav"] = nav_df.apply(fix_nav, axis=1)

        shutil.copy(nav_path, nav_path + ".bak_repair")
        nav_df.to_csv(nav_path, index=False)
        print(f"✅ nav_history.csv réparé ({len(nav_df)} entrées)")
        print(f"\nDernières NAV corrigées:")
        print(nav_df.tail(5).to_string(index=False))

    print("\n🎉 Réparation terminée ! Relancez l'app Streamlit.")

if __name__ == "__main__":
    main()
