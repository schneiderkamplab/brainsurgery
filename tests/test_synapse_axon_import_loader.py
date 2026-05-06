from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonExprCall,
    AxonExprDo,
    AxonExprName,
    AxonExprPipe,
    ast_equal,
    render_axon_file,
)
from brainsurgery.synapse.axon.load import load_axon_files_from_path
from brainsurgery.synapse.axon.parse import parse_axon_program_from_path
from brainsurgery.synapse.axon.resolve import (
    resolve_axon_program_from_path,
    resolve_axon_program_to_source,
)


def _write(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def test_import_loader_warns_for_unused_qualified_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export ping

        ping :: Tensor[B,S,D] -> Tensor[B,S,D]
        ping x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = x
        """,
    )

    program = resolve_axon_program_from_path(root)
    messages = [diag.message for diag in program.diagnostics]
    assert any("unused qualified import Foo" in message for message in messages)


def test_load_stage_returns_root_and_import_closure_as_parsed_ast_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export bar

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = bar x
        """,
    )

    loaded = load_axon_files_from_path(root)
    assert loaded.root_path == root.resolve()
    assert [f.namespace for f in loaded.files] == ["Foo", None]
    assert loaded.files[0].ast.exports == ("bar",)
    assert loaded.files[1].ast.imports == ("Foo",)


def test_load_stage_detects_import_cycles(tmp_path: Path) -> None:
    _write(
        tmp_path / "A.axon",
        """
        import B

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = x
        """,
    )
    _write(
        tmp_path / "B.axon",
        """
        import A

        helper :: Tensor[B,S,D] -> Tensor[B,S,D]
        helper x = x
        """,
    )

    with pytest.raises(ValueError, match="Cyclic Axon imports detected"):
        load_axon_files_from_path(tmp_path / "A.axon")


def test_import_loader_warns_for_unused_unqualified_import_member(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export (bar, baz)

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = x

        baz :: Tensor[B,S,D] -> Tensor[B,S,D]
        baz x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar, baz)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = bar x
        """,
    )

    program = resolve_axon_program_from_path(root)
    messages = [diag.message for diag in program.diagnostics]
    assert any("unused unqualified import Foo.baz" in message for message in messages)
    assert not any("unused unqualified import Foo.bar" in message for message in messages)


def test_import_loader_links_calls_and_prunes_unreachable_imported_defs(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export bar

        helper :: Tensor[B,S,D] -> Tensor[B,S,D]
        helper x = x

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = helper x

        dead :: Tensor[B,S,D] -> Tensor[B,S,D]
        dead x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = bar x
        """,
    )

    modules = resolve_axon_program_from_path(root).modules
    names = [module.name for module in modules]

    assert "main" in names
    assert "Foo.bar" in names
    assert "Foo.helper" in names
    assert "Foo.dead" not in names

    main_module = next(module for module in modules if module.name == "main")
    assert main_module.statements == ()
    call = main_module.body_expr
    assert isinstance(call, AxonExprCall)
    assert call.callee == "Foo.bar"


def test_import_loader_keeps_pipe_stage_name_valid_for_unqualified_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export bar

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = do
          y <- x |> bar
          return y
        """,
    )

    modules = resolve_axon_program_from_path(root).modules
    main_module = next(module for module in modules if module.name == "main")
    assert main_module.body_expr is None
    stmt = main_module.statements[0]
    assert isinstance(stmt, AxonBind)
    pipe = stmt.expr
    assert isinstance(pipe, AxonExprPipe)
    assert isinstance(pipe.stages[0], AxonExprName)
    assert pipe.stages[0].name == "Foo.bar"


def test_import_loader_keeps_qualified_imported_module_reachable(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export bar

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = Foo.bar x
        """,
    )

    modules = resolve_axon_program_from_path(root).modules
    names = {module.name for module in modules}
    assert "main" in names
    assert "Foo.bar" in names


def test_resolve_renders_import_free_program(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export bar

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = bar x
        """,
    )

    resolved = resolve_axon_program_to_source(root)
    assert "import " not in resolved
    assert "Foo.bar :: Tensor[B,S,D] -> Tensor[B,S,D]" in resolved
    assert "main :: Tensor[B,S,D] -> Tensor[B,S,D]" in resolved


def test_resolve_renders_valid_inline_do_branches(tmp_path: Path) -> None:
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Cache

        main :: Tensor[B,H,T,DH] -> Tensor[B,H,T,DH]
        main x = do
          k, _, _ <- Cache.update null x x
          return k
        """,
    )

    resolved = resolve_axon_program_to_source(root)
    out = tmp_path / "resolved.axon"
    out.write_text(resolved, encoding="utf-8")

    reparsed = parse_axon_program_from_path(out)
    report = resolve_axon_program_from_path(root)

    assert "import " not in resolved
    assert "Cache.update ::" in resolved
    assert "?CacheLayer[" in resolved
    assert ast_equal(reparsed, report.ast)
    assert render_axon_file(reparsed) == resolved


def test_resolve_strict_fails_on_warnings(tmp_path: Path) -> None:
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        dead :: Tensor[B,S,D] -> Tensor[B,S,D]
        dead x = x

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = x
        """,
    )

    with pytest.raises(ValueError, match="strict mode failed on warnings"):
        resolve_axon_program_from_path(root, strict=True)


def test_resolve_prunes_unused_root_definitions_and_reports_warning(tmp_path: Path) -> None:
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        helper :: Tensor[B,S,D] -> Tensor[B,S,D]
        helper x = x

        dead :: Tensor[B,S,D] -> Tensor[B,S,D]
        dead x = x

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = helper x
        """,
    )

    program = resolve_axon_program_from_path(root)
    names = {module.name for module in program.modules}
    assert names == {"helper", "main"}
    assert any(diag.message == "unused definition dead" for diag in program.diagnostics)


def test_resolve_reports_unused_imported_definitions_with_source_path(tmp_path: Path) -> None:
    foo = tmp_path / "Foo.axon"
    _write(
        foo,
        """
        export bar

        helper :: Tensor[B,S,D] -> Tensor[B,S,D]
        helper x = x

        bar :: Tensor[B,S,D] -> Tensor[B,S,D]
        bar x = helper x

        dead :: Tensor[B,S,D] -> Tensor[B,S,D]
        dead x = x
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (bar)

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = bar x
        """,
    )

    program = resolve_axon_program_from_path(root)
    dead_diag = next(
        diag for diag in program.diagnostics if diag.message == "unused definition Foo.dead"
    )
    assert dead_diag.file_path == foo


def test_resolve_keeps_constant_definitions_without_folding(tmp_path: Path) -> None:
    _write(
        tmp_path / "Foo.axon",
        """
        export P

        P = 1 + 2
        """,
    )
    root = tmp_path / "root.axon"
    _write(
        root,
        """
        import Foo (P)

        POS = P + 1

        main :: Tensor[B,S,D] -> Tensor[B,S,D]
        main x = do
          y <- scope@{POS} do
            return x
          return y
        """,
    )

    resolved = resolve_axon_program_to_source(root)
    assert "__Foo__P = 1 + 2" in resolved
    assert "POS = __Foo__P + 1" in resolved
    assert "__Foo__P = 3" not in resolved
    assert "POS = 4" not in resolved
    out = tmp_path / "resolved.axon"
    out.write_text(resolved, encoding="utf-8")
    parse_axon_program_from_path(out)
