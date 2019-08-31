# handwriting
직접 쓴 손글씨를 파일로 만들어 판별하는 딥러닝 코드입니다.
기본적으로 https://tykimos.github.io/2017/06/10/CNN_Data_Augmentation/ 블로그를 기반으로 인공신경망을 제 글씨체에 맞게 설계 해보았습니다.
블로그의 저자분의 권유대로 직접 데이터를 만듦으로써 가공되지 않은 상태에서 실제파일을 전처리하는데 꽤 힘든부분이 많았습니다.
필요하신 분들은 'handwrite.py' 파일중에 23행과 31행의 ''안에 파일경로만 본인컴퓨터의 해당파일 경로로 바꾸어서 코드를 실행하시면 될 것입니다.
23행은 훈련용 데이터이며 31행은 테스트 데이터 입니다.



Deep learning code by handwriting handwriting into a file.
Basically, based on the blog https://tykimos.github.io/2017/06/10/CNN_Data_Augmentation/ I created my own handwriting of artificial neural networks.
You can do as recommended by anyone in the content of the blog.
For those who need it can only be the `` internal file path '' at line 23 and line 31 of the 'handwrite.py' file.
Line 23 is for training data.
