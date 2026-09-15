"""Windows 환경 호환 shim — 반드시 pandas/streamlit보다 먼저 임포트할 것.

이 컴퓨터의 Windows 애플리케이션 제어 정책이 pyarrow의 네이티브 계산 모듈
(`pyarrow._compute`)을 차단한다. pandas는 pyarrow가 설치돼 있으면(버전만
확인하고 예외처리 없이) 무조건 `import pyarrow.compute`를 시도하므로, 그대로
두면 `import pandas` 자체가 실패한다.

우리 앱은 ArrowDtype 확장 컬럼(.list/.struct 접근자 등)을 전혀 쓰지 않으므로,
`pyarrow.compute`가 막혀 있을 때만 빈 더미 모듈로 등록해서 pandas의 import를
통과시킨다. pyarrow 자체(Streamlit이 데이터프레임 직렬화에 쓰는 `pyarrow.Table`
등)는 차단 대상이 아니라서 실제 모듈을 그대로 쓴다 — 이 shim은 `.compute`
서브모듈 하나만 건드린다.

정상 환경(정책이 없는 다른 컴퓨터 등)에서는 진짜 `pyarrow.compute`가 그냥
임포트되고 이 shim은 아무 일도 하지 않는다.
"""
from __future__ import annotations

import sys
import types

class _DummyComputeModule(types.ModuleType):
    """pandas의 `pandas/core/arrays/arrow/array.py`가 모듈 로드 시점에
    `pc.equal`, `pc.less` 같은 함수들을 딕셔너리 값으로 즉시 참조한다
    (지연 호출이 아님). 그래서 빈 모듈로는 부족하고, 어떤 이름을 물어봐도
    "호출되면 에러를 내는 더미 함수"를 돌려줘야 import 자체가 통과한다.
    이 더미 함수들은 우리 앱이 ArrowDtype 컬럼을 쓰지 않는 한 실제로
    호출되지 않는다.
    """

    def __getattr__(self, name: str):
        # dunder(다른 라이브러리·inspect 모듈이 `__file__`, `__loader__` 같은
        # 걸 찾아볼 때 쓰는 이름)는 진짜로 "없음"을 알려줘야 한다. 여기서도
        # 더미 함수를 돌려주면 Streamlit 등 다른 코드의 모듈 내부 검사 로직이
        # 문자열 대신 함수를 받아서 엉뚱하게 깨진다(실제로 겪은 버그).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _unavailable(*args, **kwargs):
            raise RuntimeError(
                f"pyarrow.compute.{name}은 이 컴퓨터에서 쓸 수 없습니다 "
                "(Windows 정책이 pyarrow 네이티브 계산 모듈을 차단함). "
                "이 앱은 pandas ArrowDtype 기능을 쓰지 않으므로 정상 동작 "
                "중에는 호출되지 않아야 합니다."
            )

        return _unavailable


if "pyarrow.compute" not in sys.modules:
    try:
        import pyarrow.compute  # noqa: F401  # 정상 환경이면 진짜 모듈을 그대로 씀
    except ImportError:
        import pyarrow  # 이건 성공해야 한다 — 차단 대상은 .compute뿐

        _stub = _DummyComputeModule("pyarrow.compute")
        sys.modules["pyarrow.compute"] = _stub
        pyarrow.compute = _stub
