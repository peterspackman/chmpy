import pyparsing as pp
from pyparsing import Literal as Lit
from pyparsing import Optional as Opt
from pyparsing import Regex, oneOf
from chmpy.core.element import Element
import numpy as np

pp.ParserElement.enablePackrat()


class SMILESParser:
    def __init__(self):
        OrganicSymbol = Regex("Br?|Cl?|N|O|P|S|F|I|At|Ts|b|c|n|o|p|s").setParseAction(
            self.parse
        )
        Symbol = Regex(
            "A(c|g|l|m|r|s|t|u)|"
            "B(a|e|h|i|k|r)?|"
            "C(a|d|e|f|l|m|n|o|r|s|u)?|"
            "D(b|s|y)|"
            "E(r|s|u)|"
            "F(e|l|m|r)?|"
            "G(a|d|e)|"
            "H(e|f|g|o|s)?|"
            "I(n|r)?|"
            "Kr?|"
            "L(a|i|r|u|v)?|"
            "M(c|g|n|o|t)?|"
            "N(a|b|d|e|h|i|o|p)?|"
            "O(g|s)?|"
            "P(a|b|d|m|o|r|t|u)?|"
            "R(a|b|e|f|g|h|n|u)|"
            "S(b|c|e|g|i|m|n|r)?|"
            "T(a|b|c|e|h|i|l|m|s)|"
            "U|V|W|Xe|Yb?|Z(n|r)|"
            "b|c|n|o|p|se?|as"
        )
        Chiral = Regex("@@?")
        Fifteen = Regex("1(0|1|2|3|4|5)|2|3|4|5|6|7|8|9")
        Charge = pp.MatchFirst(
            [
                Lit("+") + Opt(pp.MatchFirst([Lit("+"), Fifteen])),
                Lit("-") + Opt(pp.MatchFirst([Lit("-") ^ Fifteen])),
            ]
        ).setParseAction(lambda x: int(x[0]))
        HCount = Regex("H[0-9]?")
        Isotope = Regex("[0-9]?[0-9]?[0-9]")
        Map = Regex(":[0-9]?[0-9]?[0-9]")
        Dot = Lit(".").setParseAction(self._parse_dot)
        Bond = Regex(r"-|=|#|$|\\").setParseAction(self._parse_bond)
        RNum = Regex("[0-9]|(%[0-9][0-9])").setParseAction(self._parse_ring_num)
        Line = pp.Forward()
        Atom = pp.Forward()
        LBr = Lit("(").setParseAction(self._parse_lbr)
        RBr = Lit(")").setParseAction(self._parse_rbr)
        Branch = LBr + (pp.OneOrMore(Opt(pp.MatchFirst([Bond, Dot])) + Line)) + RBr
        Chain = pp.OneOrMore(
            pp.MatchFirst(
                [
                    (Dot + Atom("atoms")),
                    (
                        Opt(Bond("explicit_bonds"))
                        + pp.MatchFirst([Atom("atoms"), RNum("rings")])
                    ),
                ]
            )
        )
        BracketAtom = Lit("[") + Opt(Isotope("isotopes")) + Symbol + Opt(
            Chiral("chirality")
        ) + Opt(HCount("explicit_hydrogens")) + Opt(Charge) ^ Opt(Map) + Lit("]")
        Atom << pp.MatchFirst([OrganicSymbol, BracketAtom]).setParseAction(
            self._parse_atom
        )
        Line << Atom("atom") + pp.ZeroOrMore(pp.MatchFirst([Chain, Branch]))
        self.parser = Line

    def parse(self, tok):
        self.count += 1

    def _parse_atom(self, tok):
        N = len(self.atoms)
        if self._prev_idx > -1:
            self.bonds.append((self._prev_idx, N + 1, self._bond_type))
        self._bond_type = "-"
        self.atoms.append(tok[0])
        self._prev_idx = N + 1

    def _parse_dot(self, tok):
        self._prev_idx = -1

    def _parse_bond(self, tok):
        self._bond_type = tok[0]

    def _parse_ring_num(self, toks):
        idx = int(toks[0])
        if idx in self._rings:
            a, b = self._rings[idx][0], len(self.atoms)
            self._rings[idx] = (a, b)
            self.bonds.append((a, b, "r"))
        else:
            self._rings[idx] = (len(self.atoms), -1)

    def _parse_lbr(self, toks):
        self._branch_idx = self._prev_idx

    def _parse_rbr(self, toks):
        self._prev_idx = self._branch_idx
        self._branch_idx = -1

    def parseString(self, s):
        self.count = 0
        self._prev_idx = -1
        self._branch_idx = -1
        self.atoms = []
        self.bonds = []
        self._bond_type = "-"
        self._rings = {}
        self.parser.parseString(s)
        return (self.atoms, self.bonds)


_DEFAULT_PARSER = SMILESParser()


def parse(s):
    return _DEFAULT_PARSER.parseString(s)
