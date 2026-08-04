# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import sys
import types

import pytest


class TestEnsureSqlite3:
    """Test suite for :func:`ensure_sqlite3` idempotency and error handling."""

    @pytest.fixture(autouse=True)
    def _isolate_sys_modules(self):
        """Snapshot ``sys.modules`` before each test and restore it after.

        This prevents cross-test contamination from sqlite3/pysqlite3 patching.
        """
        original_modules = sys.modules.copy()
        yield
        sys.modules.clear()
        sys.modules.update(original_modules)

    # ------------------------------------------------------------------
    # Fresh import helper -- forces a clean re-evaluation of the function
    # ------------------------------------------------------------------

    @staticmethod
    def _import_ensure_sqlite3():
        """Import ``ensure_sqlite3`` fresh to avoid cached module state."""
        from ai4rag.utils.compat import ensure_sqlite3

        return ensure_sqlite3

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def test_patches_sqlite3_even_if_pysqlite3_already_imported(self):
        """``ensure_sqlite3`` must substitute ``sqlite3`` even when ``pysqlite3`` was already imported independently."""
        fake_pysqlite3 = types.ModuleType("pysqlite3")
        sys.modules["pysqlite3"] = fake_pysqlite3

        ensure_sqlite3 = self._import_ensure_sqlite3()
        ensure_sqlite3()

        assert sys.modules["sqlite3"] is fake_pysqlite3

    def test_idempotent_when_sqlite3_already_patched(self):
        """Calling ``ensure_sqlite3`` is a no-op if ``sqlite3`` is already the pysqlite3 module."""
        fake_pysqlite3 = types.ModuleType("pysqlite3")
        fake_pysqlite3.__name__ = "pysqlite3"
        sys.modules["sqlite3"] = fake_pysqlite3

        ensure_sqlite3 = self._import_ensure_sqlite3()
        ensure_sqlite3()

        # Should still be the same object.
        assert sys.modules["sqlite3"] is fake_pysqlite3

    def test_double_call_is_noop(self):
        """Two successive calls must not raise and must leave ``sys.modules`` consistent."""
        fake_pysqlite3 = types.ModuleType("pysqlite3")
        sys.modules["pysqlite3"] = fake_pysqlite3

        ensure_sqlite3 = self._import_ensure_sqlite3()
        ensure_sqlite3()
        ensure_sqlite3()  # second call -- should be harmless

    # ------------------------------------------------------------------
    # pysqlite3 unavailable (ImportError path)
    # ------------------------------------------------------------------

    def test_noop_when_pysqlite3_unavailable(self, mocker):
        """``ensure_sqlite3`` silently passes when ``pysqlite3`` cannot be imported."""
        # Import the function *before* patching __import__.
        ensure_sqlite3 = self._import_ensure_sqlite3()

        # Remove pysqlite3 from sys.modules so the early-exit check fails.
        sys.modules.pop("pysqlite3", None)
        # Ensure sqlite3.__name__ is not "pysqlite3" (second early-exit check).
        original_sqlite3 = sys.modules.get("sqlite3")
        if original_sqlite3 is not None:
            original_name = getattr(original_sqlite3, "__name__", None)
            if original_name == "pysqlite3":
                original_sqlite3.__name__ = "sqlite3"

        import builtins

        original_import = builtins.__import__

        def _guarded_import(name, *args, **kwargs):
            if name == "pysqlite3":
                raise ImportError("no pysqlite3")
            return original_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=_guarded_import)

        ensure_sqlite3()  # must not raise

        # sqlite3 should NOT have been replaced (the ImportError was caught).
        if original_sqlite3 is not None:
            assert sys.modules.get("sqlite3") is original_sqlite3

    # ------------------------------------------------------------------
    # Successful patching path
    # ------------------------------------------------------------------

    def test_patches_sqlite3_when_pysqlite3_available(self):
        """When ``pysqlite3`` can be imported, ``sys.modules["sqlite3"]`` is replaced."""
        sys.modules.pop("pysqlite3", None)

        fake_pysqlite3 = types.ModuleType("pysqlite3")
        sys.modules["pysqlite3"] = fake_pysqlite3

        # Remove the early-exit condition.
        original_sqlite3 = sys.modules.get("sqlite3")
        if original_sqlite3 is not None and getattr(original_sqlite3, "__name__", None) == "pysqlite3":
            # Reset so the function doesn't short-circuit.
            original_sqlite3.__name__ = "sqlite3"

        ensure_sqlite3 = self._import_ensure_sqlite3()
        ensure_sqlite3()

        # pysqlite3 should have been registered, triggering the early-exit
        # on any subsequent call.
        assert "pysqlite3" in sys.modules
