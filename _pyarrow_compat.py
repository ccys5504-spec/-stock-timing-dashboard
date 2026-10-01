"""Windows 환경 호환 shim — 반드시 pandas/streamlit보다 먼저 임포트할 것.

이 컴퓨터의 Windows 애플리케이션 제어 정책이 pyarrow의 네이티브 계산 모듈
(`pyarrow._compute`)을 차단했었다. pandas가 pyarrow 설치를 감지하면 무조건
`import pyarrow.compute`를 시도하는 버전에서는 그대로 두면 `import pandas`
자체가 실패했고, 그걸 막으려고 `.compute`만 빈 더미 모듈로 등록해왔다.

2026-10-01: 정책이 더 세져서 `pyarrow.compute`뿐 아니라 `pyarrow` 패키지
자체(`pyarrow.lib` 네이티브 모듈)의 import조차 "DLL load failed"로 막히는 걸
확인했다. 이전 가정("bare `import pyarrow`는 항상 성공한다")이 깨진 것이다.
그런데 동시에 확인한 사실: **지금 고정해 쓰는 pandas==2.3.3은 애초에
pyarrow를 import 시점에 요구하지 않는다** — `import pandas`만 단독으로
실행해보면 pyarrow 없이도 아무 문제 없이 끝난다. 즉 이 shim이 원래 막으려던
문제(구버전 pandas의 즉시 probe)가 지금 버전에는 해당하지 않을 가능성이 커서,
pyarrow를 아예 못 구하는 환경에서는 **아무것도 하지 않고 조용히 넘어가는 쪽이
더 안전**하다(예전처럼 여기서 예외를 던지면 앱 전체가 시작도 못 한다).

우리 앱은 ArrowDtype 확장 컬럼(.list/.struct 접근자 등)을 전혀 쓰지 않으므로,
`pyarrow.compute`가 막혀 있을 때만 빈 더미 모듈로 등록해서 pandas의 import를
통과시킨다. pyarrow 자체(Streamlit이 데이터프레임 직렬화에 쓰는 `pyarrow.Table`
등)가 살아있으면 그대로 쓴다 — 이 shim은 가능하면 `.compute` 서브모듈 하나만
건드리고, 그조차 안 되면 손대지 않는다.

⚠️ pyarrow가 정말 전혀 동작하지 않는 환경에서는, Streamlit이 데이터프레임을
화면에 그릴 때(`st.dataframe`, `st.data_editor` 등) pyarrow를 실제로 필요로
할 수 있어 그 호출 시점에 별도의 오류가 날 수 있다 — 이 shim이 막는 건 어디
까지나 "import 시점에 앱 전체가 죽는 것"까지이고, 그 문제 자체(Windows 정책이
pyarrow 네이티브 모듈을 막는 것)는 파이썬 코드로 고칠 수 없다.

정상 환경(정책이 없는 다른 컴퓨터 등)에서는 진짜 `pyarrow.compute`가 그냥
임포트되고 이 shim은 아무 일도 하지 않는다.
"""
from __future__ import annotations

import sys
import types
import warnings

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
        try:
            import pyarrow  # 예전엔 이게 항상 성공했다 — 차단 대상은 .compute뿐이었음
        except ImportError as e:
            # 2026-10-01: pyarrow 패키지 자체가 막힌 환경. 더미를 등록할 실제 pyarrow 모듈이
            # 없으니 등록은 포기하고 경고만 남긴다 — pandas==2.3.3은 이것 없이도 동작하는
            # 것을 확인했다. st.dataframe() 등 pyarrow를 실제로 쓰는 호출은 그 시점에 따로
            # (더 구체적인) 오류를 낼 것이다.
            warnings.warn(
                f"pyarrow를 전혀 import할 수 없습니다({type(e).__name__}: {e}). "
                "pandas는 계속 쓸 수 있지만, 데이터프레임을 화면에 그리는 기능(st.dataframe 등)이 "
                "이후에 별도로 실패할 수 있습니다.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            _stub = _DummyComputeModule("pyarrow.compute")
            sys.modules["pyarrow.compute"] = _stub
            pyarrow.compute = _stub
