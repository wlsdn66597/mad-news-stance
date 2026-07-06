"""Task 인터페이스 — debate 엔진이 task 를 모르게 하기 위한 얇은 추상화.

각 task 는 데이터 로딩 / 질문 프롬프트 / 답 파싱 / 채점을 제공한다.
Phase 1(GSM8K, MMLU)과 Phase 2(stance)가 같은 method/debate 코드를 재사용한다.
"""


class Task:
    name = "base"

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
