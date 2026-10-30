#!/usr/bin/env python3
"""The Beta card and the PXA Coins line under licd's Valued-only beta, and the v3.1 / beta 1 handover line.
No network, no browser (tests/beta-wording-gui-check.py is the browser half)."""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pxa_control import routes as R  # noqa: E402

UI = os.path.join(ROOT, "tools", "pxa_control_ui")
HANDOVER = "v3.1 / beta 1 users: run /beta in Discord."
BANNED = re.compile(r"\b(paid|sell|sold|buy|bought|purchase|price|pricing)\b", re.I)


def read(name):
    with open(os.path.join(UI, name), encoding="utf-8") as f:
        return f.read()


class BetaCard(unittest.TestCase):
    def test_valued_only_reaches_the_card(self):
        v = {"wallet_id": "PXW-AAAA-BBBB-CCCC", "coins": 0, "beta": {"release": None, "access": False, "how": None, "valued_only": True}}
        c = R._beta_card(v)
        self.assertEqual((c["valued_only"], c["release"], c["access"], c["cost"], c["short_by"]), (True, None, False, 0, 0))

    def test_old_licd_answer_is_not_valued_only(self):
        v = {"wallet_id": "PXW-AAAA-BBBB-CCCC", "coins": 10, "beta": {"release": "4.0-beta.2", "access": False, "coins": 50}}
        c = R._beta_card(v)
        self.assertEqual((c["valued_only"], c["release"], c["cost"], c["short_by"]), (False, "4.0-beta.2", 50, 40))


class Wording(unittest.TestCase):
    def test_beta_card_has_the_handover_line_and_valued_only_text(self):
        s = read("index.html")
        m = re.search(r'<p[^>]*id="bc-v31"[^>]*>([^<]*)</p>', s)
        self.assertTrue(m and HANDOVER in m.group(1))
        body = s[s.index("function renderBeta()"):s.index("async function loadBeta()")]
        self.assertIn("B.valued_only", body)
        self.assertIn("Valued Supporter reward and spends no coins", body)
        self.assertNotIn("Beta builds are a thank-you for supporters", body)
        self.assertIn("/beta", body)                     # /beta is the door, not /key
        self.assertNotIn("/key", body)

    def test_coins_panel_line_has_the_handover(self):
        s = read("wallet.js")
        body = s[s.index("function betaLine(w)"):s.index("function betaState(w)")]
        self.assertIn(HANDOVER, body)
        self.assertIn("Beta access: for Valued Supporters (no coins)", s)

    def test_no_banned_words_in_the_new_strings(self):
        for name in ("index.html", "wallet.js"):
            s = read(name)
            for line in s.splitlines():
                if "handover" in line or "valued_only" in line or "Valued Supporter reward" in line:
                    self.assertIsNone(BANNED.search(line), line)


if __name__ == "__main__":
    unittest.main()
