"""Task 인터페이스 — debate 엔진이 task 를 모르게 하기 위한 얇은 추상화.

각 task 는 데이터 로딩 / 질문 프롬프트 / 답 파싱 / 채점 + debate 프롬프트(언어별)를 제공한다.
"""


class Task:
    name = "base"
    debate_template = None   # task 가 언어/포맷에 맞게 제공; None 이면 debate 엔진 기본값 사용

    def load(self, split, n, seed=0):
        """[{'id', 'gold', ...}] 형태의 아이템 리스트를 반환."""
        raise NotImplementedError

    def question(self, item, style="cot"):
        """모델에 줄 질문 문자열. style: 'vanilla'(직답) | 'cot'(단계별 추론)."""
        raise NotImplementedError

    def parse(self, text):
        """모델 출력에서 정규화된 예측 답을 추출(실패 시 None)."""
        raise NotImplementedError

    def correct(self, pred, item):
        """예측이 정답(item['gold'])과 맞는지."""
        raise NotImplementedError
