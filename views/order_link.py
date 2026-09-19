"""키움증권 WTS(웹 트레이딩)로 바로 이동하는 바로가기.

이 앱은 주문을 대신 넣지 않는다 — 오직 키움 WTS를 새 탭으로 열어주고,
종목코드를 한 번에 복사할 수 있게만 해준다. 로그인·주문은 전부 사용자가
키움 화면에서 직접 한다(계정 정보나 주문 권한이 이 앱을 거치지 않음).

WTS를 이 앱 안에 iframe으로 끼워 넣지 않은 이유: (1) 증권사 사이트는 보통
다른 사이트 안에 넣는 것 자체를 막아두고(클릭재킹 방지), (2) 넣을 수 있다
해도 남의 화면 안에서 로그인/주문을 하는 건 보안상 좋지 않다.

키움 WTS가 종목코드를 URL로 받는 공식 방법은 확인하지 못해서(로그인이
필요한 화면이라 검증 불가) 임의로 URL을 만들어 붙이지 않았다 — 대신 종목
코드 복사 칸을 붙여둔다.
"""
from __future__ import annotations

import streamlit as st

KIWOOM_WTS_URL = "https://wts2.kiwoom.com/"


def render_order_link(name: str | None = None, code: str | None = None) -> None:
    """키움 WTS 새 탭 열기 버튼(+ 종목코드 복사 칸)을 그린다.

    한 화면에 여러 번 부르면 같은 버튼이 중복돼서 에러가 나므로, 뷰마다
    한 번만 호출한다.
    """
    cols = st.columns([1.4, 1, 2.6]) if code else st.columns([1.4, 3.6])
    cols[0].link_button(
        "🏦 키움 WTS에서 주문하기", KIWOOM_WTS_URL, use_container_width=True,
        help="키움증권 웹 트레이딩(wts2.kiwoom.com)을 새 탭으로 엽니다.",
    )
    if code:
        with cols[1]:
            # st.code는 마우스를 올리면 복사 버튼이 나온다 — WTS의 종목코드 칸에 붙여넣기용
            st.code(code, language=None)
        cols[2].caption(
            f"← {name or '이 종목'} 종목코드(복사해서 WTS에 붙여넣기). "
            "이 앱은 주문을 대신 넣지 않으니, 로그인과 주문은 키움 화면에서 직접 하세요."
        )
    else:
        cols[1].caption(
            "이 앱은 주문을 대신 넣지 않습니다. 로그인과 주문은 키움 화면에서 직접 하세요."
        )
