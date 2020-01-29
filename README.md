# handwriting
직접 쓴 손글씨를 파일로 만들어 판별하는 딥러닝 코드입니다.
CNN 인공신경망을 제 글씨체에 맞게 설계 해보았습니다.
직접 데이터를 만듦으로써 가공되지 않은 실제파일을 전처리하였습니다.
필요하신 분들은 'handwrite.py' 파일중에 23행과 31행의 
' '안에 파일경로만 본인컴퓨터의 해당파일 경로로 바꾸어서 코드를 실행하시면 될 것입니다.
23행은 훈련용 데이터이며 31행은 테스트 데이터 입니다.

테스트 결과 약 80%정확도로 손글씨 예측에 성공률을 보였습니다.
기본적인 CNN신경망을 설계한 것으로 추후에 추가적인 튜닝과정을 거치면 성능향상을 기대해 볼 수 있을 것 같습니다.

결과값의 해석은 아래와 같습니다.
각행의 위치별로 인덱스[0] = 원, 인덱스[1] = 사각형, 인덱스[2] = 삼각형 을 나타냅니다.
각행에서 가장 큰값으로 계산된 위치가 CNN신경망이 정답으로 예측한 도형입니다.



<br>

<img width="1096" alt="스크린샷 2020-01-29 오후 11 11 34" src="https://user-images.githubusercontent.com/45910733/73365519-bdf83a00-42ef-11ea-897c-c72f47097ac0.png">
