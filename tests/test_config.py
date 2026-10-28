import unittest

from config import _parse_star_packs


class ConfigParsingTests(unittest.TestCase):
    def test_parses_star_packs(self):
        self.assertEqual(_parse_star_packs("100000:50, 500000:200"), [(100000, 50), (500000, 200)])

    def test_rejects_malformed_or_nonpositive_packs(self):
        for value in ("100:2:3", "abc:3", "100:0", "-1:10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_star_packs(value)

    def test_limits_package_count(self):
        with self.assertRaises(ValueError):
            _parse_star_packs(",".join(["1:1"] * 11))


if __name__ == "__main__":
    unittest.main()
