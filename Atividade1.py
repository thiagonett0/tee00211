import pandas as pd
import numpy as np
from scipy import linalg


def imprimeMatriz(A):

    """
    Descrição: imprime matriz

    Entrada(s):
                i) A (list[list[complex]]): matriz a ser impressa no terminal;

    Saída(s):
                i)
    """

    print('\n')
    for row in A:
        for col in row:
            print("{:13.2f}".format(col), end=" ")
        print("")
    print('\n')
    return


dictbarra = {
            'Id.':['Barra 1', 'Barra 2', 'Barra 3',
                    'Barra 4', 'Barra 5', 'Barra 6', 'Barra 7'],
            'Num.': [1, 2, 3, 4, 5, 6, 7],
            'Area':[1, 1, 1, 1, 1, 1, 1],
            'VBase': [13.8, 230, 230, 230, 230, 230, 13.8],
            'SBase': [100, 100, 100, 100, 100, 100, 100],
            'V (pu)': [1.0, 1.0, 1.0, 1.0, 1, 1, 1.087],
            'Theta (rad)': [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0],
            'P (MW)': [0, -50, -50, -50, -50, -50, 180],
            'Q (Mvar)': [0, -30, -30, -30, -30, -30, 0]
            }


dictlinha = lambda L2: \
            {
            'Nome':['L1', 'L2', 'L3', 'L4', 'L5'],
            'Num.': [1, 2, 3, 4, 5],
            'Circuito': [1, 1, 1, 1, 1],
            'Area':[1, 1, 1, 1, 1],
            'De': [2, 3, 4, 5, 4],
            'Para': [3, 4, 5, 6, 6],
            'Comprimento (km)': [15, L2, 40, 15, 50],
            'Z0 (Ohm/km)': [0.2+0.15*1j]*5,
            'Z1 (Ohm/km)': [0.08+0.5*1j]*5,
            'Y1 (S/km)': [3.3*1e-6*1j]*5
            }


dicttrafo = {'Nome':['T1', 'T2'],
            'Num.': [1, 2],
            'Circuito': [1, 1],
            'Area':[1, 1],
            'Primario': [1, 7],
            'Secundario': [2, 6],
            'S (MVA)': [100, 200],
            'R1 (pu)': [0.0, 0.0],
            'X1 (pu)': [0.1, 0.1],
            'R0 (pu)': [0.0, 0.0],
            'X0 (pu)': [0.1, 0.1],
            'Conexão Prim.': ['D', 'D'],
            'Conexão Sec.': ['Yg','Yg'],
            'Tap': [1, 1]
            }


dictger = {
            'Nome':['G1', 'G2'],
            'Num.': [1, 2],
            'Area':[1, 1],
            'Barra': [1, 7],
            'S (MVA)': [100, 200],
            'R1 (pu)': [0.0, 0.0],
            'Xd" (pu)': [0.12, 0.12],
            'R0 (pu)': [0.0, 0.0],
            'X0 (pu)': [0.05, 0.05],
            'Rn (pu)': [0.0, 0.0],
            'Xn (pu)': [0, 0],
            'Conexão': ['Yg', 'Yg'],
            'Unidades': [1, 1]
            }


def fortescue(v012):
    
    """
    Descrição: calcula o vetor em fase a partir do vetor de sequência;

    Entrada(s):
                i) v012 (np.array): vetor em sequência;
    
    Saída(s):
                i) vabc (np.array): vetor em fases.
    """

    vabc = v012.copy()
    a = np.exp(1j*2*np.pi/3)
    for i in range(int(len(v012)/3)):
        vabc[i:i+3] = np.matmul([[1, 1, 1], [1, a, a**2], [1, a**2, a]], np.array(v012)[i:i+3])
    return vabc


def parametrosBarras(dictBarras):

    """
    Descrição: função que ajusta os parâmetros das barras. Nesse caso, como já estão ajustados conforme esperado, apenas os retorna;

    Entrada(s):
                i) dictBarras (dict[list]): dicionário contendo informações das barras do sistema;

    Saída(s):
                i) _
    """

    return dictBarras


def parametrosGeradores(dictGeradores):

    """
    Descrição: função que ajusta os parâmetros das barras. Nesse caso, calcula-se as quantidades em pu% por meio da subfunção puPercentual;

    Entrada(s):
                i) dictGeradores (dict[list]): dicionário contendo informações dos geradores do sistema;

    Saída(s):
                i) _
    """

    def puPercentual(v): 
        
        """
        Descrição: calcula pu% por meio de iteração em cada elemento do vetor v e multiplica por 100.
        """
        
        return list(map(lambda vi: vi*100, v))

    dictGeradores['R1 (pu%)'], dictGeradores['Xd" (pu%)'] = puPercentual(dictGeradores['R1 (pu)']), puPercentual(dictGeradores['Xd" (pu)'])
    dictGeradores['R0 (pu%)'], dictGeradores['X0 (pu%)'] = puPercentual(dictGeradores['R0 (pu)']), puPercentual(dictGeradores['X0 (pu)'])
    dictGeradores['Rn (pu%)'], dictGeradores['Xn (pu%)'] = puPercentual(dictGeradores['Rn (pu)']), puPercentual(dictGeradores['Xn (pu)'])
    return dictGeradores


def parametrosTransformadores(dictTrafos):

    """
    Descrição: calcula os parâmetros de transformadores necessários para preencher no EditCepel. Para tal,

                i) determina-se a função auxiliar para cálculo da reatância em pu%;
                ii) calcula a potência aparente base;
                iii) mapeia as listas relacionadas de quantidades de sequência zero para suas respectivas saídas, calculadas
                    pelas funções;
                iv) repete o processo de forma análoga para as quantidades de sequência positiva;
    
    Entrada(s):
                i) dictTrafos (dict[list]):
    
    Saída(s):
                i) dictTrafos (dict[list]): dicionário atualizado.
    """

    def reatanciaPupor(Xpu, S):
        return list(map(lambda x, s: x*10000/s, Xpu, S))
    

    Rpu0, Xpu0 = dictTrafos['R0 (pu)'], dictTrafos['X0 (pu)']
    Rpu1, Xpu1 = dictTrafos['R1 (pu)'], dictTrafos['X1 (pu)']
    S = dictTrafos['S (MVA)']
    dictTrafos['R0 (pu%)'], dictTrafos['X0 (pu%)'] = reatanciaPupor(Rpu0, S), reatanciaPupor(Xpu0, S)
    dictTrafos['R1 (pu%)'], dictTrafos['X1 (pu%)'] = reatanciaPupor(Rpu1, S), reatanciaPupor(Xpu1, S)
    return dictTrafos


def parametrosLinha(dictLinhas, Sbase, Vbase):

    """
    Descrição: calcula os parâmetros de linha necessários para preencher no EditCepel. Para tal,

                i) determina-se as funções auxiliares para cálculo das resistências, reatâncias e susceptâncias em pu %;
                ii) calcula a impedância base para as linhas;
                iii) calcula a susceptância base para as linhas;
                iv) mapeia as listas relacionadas de quantidades de sequência zero para suas respectivas saídas, calculadas
                    pelas funções;
                v) repete o processo de forma análoga para as quantidades de sequência positiva;
    
    Entrada(s):
                i) dictLinhas (dict[list]):
                ii) Sbase (float):
                iii) Vbase (float):
    
    Saída(s):
                i) dictLinhas (dict[list]): dicionário atualizado.
    """

    def resistenciaKm(Zpu, comp):

        """
        Descrição: calcula a resistência em Ohm/km %, por meio da iteração de cada elemento do vetor v, mapeia-o conforme. 
        """

        return list(map(lambda z, c: z.real*c*100, Zpu, comp))
    
    def reatanciaKm(Zpu, comp):

        """
        Descrição: calcula a reatância em Ohm/km %, por meio da iteração de cada elemento do vetor v, mapeia-o conforme. 
        """

        return list(map(lambda z, c: z.imag*c*100, Zpu, comp))
    
    def susceptanciaKm(Bpu, comp):

        """
        Descrição: calcula a susceptância em Ohm/km %, por meio da iteração de cada elemento do vetor v, mapeia-o conforme. 
        """

        return list(map(lambda b, c: b.imag*c*100, Bpu, comp))


    Zbase = Vbase**2/Sbase
    Bbase = 1/Zbase
    Zpu0, Zpu1, Bpu, comp = np.array(dictLinhas['Z0 (Ohm/km)'])/Zbase, np.array(dictLinhas['Z1 (Ohm/km)'])/Zbase, \
                                np.array(dictLinhas['Y1 (S/km)'])/Bbase, dictLinhas['Comprimento (km)']
    dictLinhas['R0 (pu%)'], dictLinhas['X0 (pu%)'], dictLinhas['B0 (Mvar)'] = \
                            resistenciaKm(Zpu0, comp), reatanciaKm(Zpu0, comp), susceptanciaKm(Bpu, comp)
    dictLinhas['R1 (pu%)'], dictLinhas['X1 (pu%)'], dictLinhas['B1 (Mvar)'] = \
                            resistenciaKm(Zpu1, comp), reatanciaKm(Zpu1, comp), susceptanciaKm(Bpu, comp)
    return dictLinhas


def parametrosANAREDE(L2, Sbase, Vbase, dictbarra, dictlinha, dicttrafo, dictger):

    """
    Descrição: calcula os parâmetros de toda a rede necessários para preencher no EditCepel. Para tal,

                i) chama as funções de parametrização de cada parte do sistema;
    
    Entrada(s):
                i) L2 (float): tamanho da linha 2;
                ii) Sbase (float): potência aparente base para o sistema de alta tensão;
                iii) Vbase (float): tensão base para o sistema de alta tensão;
                dictLinhas (dict[list]):
                ii) Sbase (float):
                iii) Vbase (float):
    
    Saída(s):
                i) dictLinhas (dict[list]): dicionário atualizado.
    """

    dictBarras = pd.DataFrame(parametrosBarras(dictbarra))
    dictGers = pd.DataFrame(parametrosGeradores(dictger))
    dictLinhas = pd.DataFrame(parametrosLinha(dictlinha(L2), Sbase, Vbase))
    dictTrafos = pd.DataFrame(parametrosTransformadores(dicttrafo))
    print(f'Barras:\n{dictbarra}\n\nLinhas:\n{dictLinhas}\n\nTransformadores:\n{dictTrafos}\n\nGeração:\n{dictGers}')
    return dictBarras, dictLinhas, dictTrafos, dictGers


# Questão 5.2

# Considerar as ligações de geração e transformação
def yBarra(dictLinhas, dictTrafos, dictGeradores, sequencia, Zbase):

    """
    Descrição: calcula a matriz Ybarra, conforme sua lei de formação [REFERÊNCIA]. Para tal,

                i) determina-se primeiramente seu tamanho por meio do maior índice de barra em todo o sistema;
                ii) declara-se a matriz Ybarra inicialmente cheia de zeros;
                iii) altera-se suas entradas calculando as contrbuições da linhas por meio de função auxiliar;
                iv) idem anterior, porém calcula-se a contribuição dos transformadores;
                v) idem iii, porém calcula-se a contribuição dos geradores;
    
    Entrada(s):

    Saída(s):
    """

    def Ylinhas(Y, dictLinhas):

        """
        Descrição: calcula-se as admitâncias das linhas do sistema. Para tal,

                    i) declara-se as listas (vetores) de índices 'de' e 'para', comprimentos e impedâncias associadas;
                    ii) por meio de zip, itera-se sobre essas 4 características de cada linha, por vez;
                        ii.a) ajusta-se os índices das barras em conformidade com indexação do Python;
                        ii.b) assinala-se as contribuições para as respectivas entradas da Ybarra;
        
        Entrada(s):

        Saída(s):
        """

        de, para, comp, zs = dictLinhas['De'], dictLinhas['Para'], dictLinhas['Comprimento (km)'], \
                             np.array(dictLinhas[f'Z{sequencia%2} (Ohm/km)'])/Zbase
        for d, p, c, z in zip(de, para, comp, zs):
            d, p = int(d)-1, int(p)-1
            y = c/z
            Y[d][d] += y
            Y[p][p] += y
            Y[d][p] = -y
            Y[p][d] = -y
        return Y

    def Ytransformadores(Y, dictTrafos):

        """
        Descrição: calcula-se as admitâncias das linhas do sistema. Análoga à função Ylinhas.
        
        Entrada(s):

        Saída(s):
        """

        de, para, xs = dictTrafos['Primario'], dictTrafos['Secundario'], np.array(dictTrafos[f'X{sequencia%2} (pu)'])
        for d, p, x in zip(de, para, xs):
            d, p = int(d)-1, int(p)-1
            y = -1j/x
            Y[d][d] += y
            Y[p][p] += y
            Y[d][p] = -y
            Y[p][d] = -y
        return Y
    
    def Ygeradores(Y, dictGeradores):

        """
        Descrição: calcula-se as admitâncias das linhas do sistema. Análoga à função Ylinhas.
        
        Entrada(s):

        Saída(s):
        """

        barra, xs = dictGeradores['Barra'], np.array(dictGeradores[f'Xd" (pu)'])  # if sequencia !=0 else dictGeradores[f'X0 (pu)']
        for b, x in zip(barra, xs):
            b = int(b)-1
            y = -1j/x
            Y[b][b] += y
        return Y


    tamanho = max(list(dictLinhas['De']) + list(dictLinhas['Para']) + \
                  list(dictTrafos['Primario']) + list(dictTrafos['Secundario']) + list(dictGeradores['Barra']))
    Y = [[0 for _ in range(tamanho)] for _ in range(tamanho)]
    Ylinhas(Y, dictLinhas)
    Ytransformadores(Y, dictTrafos)
    Ygeradores(Y, dictGeradores)
    print(f"Ybarra: {Y}\n")
    return Y


def metodoKron():
    return



# Questão 6.1


# Questão 6.1.2
def matrizZ012(Z0, Z1, Z2):

    """
    Descrição: calcula a matriz Z012barra, que pode ser vista como mapeamentos dos elementos i,j das matrizes Z0, Z1 e Z2 para,
               respectivamente: 0) i*3+0, j*3+0; 1) i*3+1, j*3+1; 2) i*3+2, j*3+2. Para tal,

                i) declara-se a matriz Z012barra inicialmente como matriz nula;
                ii) itera-se sobre cada elemento de Z012barra;
                    ii.a) basta verificar os índices e em qual dos casos acima se encaixa;
                    ii.b) mapeia da matriz de sequência Zk para Z012barra;
    
    Entrada(s):

    Saída(s):
    """

    tamanho = int(3*len(Z0))
    Z012barra = [[0 for _ in range(tamanho)] for _ in range(tamanho)]
    for i in range(tamanho):
        for j in range(tamanho):
            if i%3==0 and j%3==0:
                Z012barra[i][j] = Z0[int(i/3)][int(j/3)]
            elif i%3==1 and j%3==1:
                Z012barra[i][j] = Z1[int(i/3)][int(j/3)]
            elif i%3==2 and j%3==2:
                Z012barra[i][j] = Z2[int(i/3)][int(j/3)]
    return Z012barra


def matrizZ012p(Z012barra, p):

    """
    Descrição: calcula a matriz Z012 na barra em que ocorreu a falta. Para tal, basta particionar a 
                matriz Z012barra em dimensões p até p+3.

    Entrada(s):
                i) Z012barra (list[list]): matriz Z barra de sequência do sistema;
                ii) p (int): barra onde ocorre a falta;

    Saída(s):
                i) Z012p (list[list]): 
    """

    Z012p = np.array(Z012barra)[p:p+3, p:p+3]
    return Z012p


# Corrigir tensão e corrente pré-falta.
# Questão 6.1.1
def condicaopreFalta(tamanho):

    """
    Descrição: determina as condições de tensão e corrente pré-falta.

    Entrada(s):
                i) tamanho (int): tamanho do sistema todo. Condição do sistema todo antes da falta.
    
    Saída(s):
                i) V (list): tensões pré-falta;
                ii) I (list): correntes pré-falta.
    """

    V, I = [1]*tamanho, [0]*tamanho
    return V, I


# Questão 6.1.3
def faltaDesequilibrada(Z012barra, p, V012preFalta, I012preFalta, tipoFalta, Za, Zb, Zc, Zg):

    """
    Descrição: calcula as tensões e correntes pós-falta para cada um dos tipos de falta. Para tal,

                i) determina Z012p e as condições pré-falta;
                ii) calcula a corrente I012f por meio de dicionário e subfunções abaixo definidas;
                iii) calcula o deltaV012f;
                iv) calcula o V012f
    """

    def faltaTri(Za, Zb, Zc, Zg):

        """
        Descrição: resolve o sistema considerando a falta trifásica.
        """

        try:
            Zf, Zfg = Za, Zg
            Z012f = np.array([[Zf+3*Zfg, 0, 0], [0, Zf, 0], [0, 0, Zf]])
            I012f = np.linalg.solve(Z012p + Z012f, V012preFalta)
            return I012f
        except np.linalg.LinAlgError:
            print('\n\nFalta trifásica deu erro! Matriz simétrica, sistema não solúvel. Resolver via expressão de curto franco\n\n')
            return np.array([np.nan]*3)

    def faltaFT(Za, Zb, Zc, Zg):
        
        """
        Descrição: resolve o sistema considerando a falta fase-terra.
        """

        try:
            Yf = 1/Za
            Y012f, U = np.ones((3, 3))*Yf/3, np.identity(3)
            I012f = np.linalg.solve(Y012f@(U+Z012p@Y012f), V012preFalta)
            return I012f
        except np.linalg.LinAlgError:
            print('\n\nFalta fase-terra deu erro! Matriz simétrica, sistema não solúvel. Resolver via expressão de curto franco\n\n')
            I012f = np.ones(3)*V012preFalta[0]/(Z012p[0][0] + Z012p[1][1] + Z012p[2][2])
            return I012f

    def faltaFF(Za, Zb, Zc, Zg):
        
        """
        Descrição: resolve o sistema considerando a falta fase-fase.
        """

        try:
            Yf = 1/Zb
            Y012f, U = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])*Yf/2, np.identity(3)
            I012f = np.linalg.solve(Y012f@(U+Z012p@Y012f), V012preFalta)
            return I012f
        except np.linalg.LinAlgError:
            print('\n\nFalta fase-fase deu erro! Matriz simétrica, sistema não solúvel. Resolver via expressão de curto franco\n\n')
            I012f = np.array([0, 1, -1])*V012preFalta[0]/(Z012p[1][1] + Z012p[2][2])
            return I012f

    def faltaFFT(Za, Zb, Zc, Zg):
        
        """
        Descrição: resolve o sistema considerando a falta fase-fase terra.
        """
        try:
            Zf, Zfg = Zb, Zg
            Y012f, U = np.array([[2*Zf, -Zf, -Zf], [-Zf, 2*Zf+3*Zfg, -Zf-3*Zfg], [-Zf, -Zf-3*Zfg, 2*Zf+3*Zfg]])/(3*(Zf*Zf+2*Zf*Zfg)), \
                       np.identity(3)
            I012f = np.linalg.solve(Y012f@(U+Z012p@Y012f), V012preFalta)
            return I012f
        except np.linalg.LinAlgError:
            I1f = V012preFalta[0]/(Z012p[1][1] + Z012p[2][2]*(Z012p[0][0] + 3*Zg)/(Z012p[0][0] + Z012p[2][2] + 3*Zg))
            I0f = - (V012preFalta[0] - Z012p[1][1]*I1f)/(Z012p[0][0] + 3*Zg)
            I2f = - (V012preFalta[0] - Z012p[1][1]*I1f)/Z012p[2][2]
            print('\n\nFalta fase-fase-terra deu erro! Matriz simétrica, sistema não solúvel. Resolver via expressão de curto franco\n\n')
            return np.array([I0f, I1f, I2f])


    Z012p, V012preFalta, I012preFalta = matrizZ012p(Z012barra, p), np.array(V012preFalta[p:p+3]), np.array(I012preFalta[p:p+3])
    dicCasos = {'trifasica': faltaTri, 'faseterra': faltaFT, 'fasefase': faltaFF, 'fasefaseterra': faltaFFT}
    I012f = dicCasos[tipoFalta](Za, Zb, Zc, Zg)
    deltaV012f = -Z012p@I012f
    V012f = deltaV012f + V012preFalta
    return V012f, I012f


# Questão 6.1.4
# Questão 6.1.5
def sistemaPosfalta(Z012, p, tipoFalta, Za, Zb, Zc, Zg):

    """
    Descrição: calcula as tensões e correntes do sistema após a ocorrência da falta. Para tal,

                i) calcula as tensões e correntes pré-falta;
                ii) calcula a tensão e corrente de falta p;
                iii) atualiza o vetor de correntes do sistema;
                iv) por meio da corrente do sistema e matriz Z, calcula a tensão do sistema;

    Entrada(s):
                i) Z012 (list[list]): matriz Zbarra de sequência;
                ii) p (int): barra onde ocorre a falta;
                iii) tipoFalta (str): tipo de falta dentre as opções: trifasica, faseterra, fasefase, fasefaseterra;
                iv) Za (complex): impedância de fase A;
                v) Zb (complex): impedância de fase B;
                vi) Zc (complex): impedância de fase C;
                vii) Zg (complex): impedância de aterramento
    
    Saída(s):
                i) V012fsistema (list): tensão do sistema após a falta.
    """

    V012preFalta, I012preFalta = condicaopreFalta(len(Z012))
    V012f, I012f = faltaDesequilibrada(Z012, p, V012preFalta, I012preFalta, tipoFalta, Za, Zb, Zc, Zg)
    I012fsistema = I012preFalta.copy()
    I012fsistema[p:p+3] = -I012f  # comentar o procedimento de todas as correntes serem 0 e só ajustar a barra p
    V012posfsistema = np.array(V012preFalta) + np.linalg.solve(Z012, I012fsistema)
    I012posfsistema = np.linalg.solve(linalg.inv(Z012), V012posfsistema) #I012fsistema = I012posfsistema
    return I012fsistema, V012posfsistema, I012posfsistema


def baseDados(Z012, p, tipoFalta, Za, Zb, Zc, Zg):

    """
    Descrição: calcula as correntes de falta, tensões pós falta e correntes pós falta do sistema. Para tal,

                i) calcula em formato ANAREDE as barras, linhas, transformadores e geradores;
                ii) determina o Zbase;
                iii) calcula as matrizes de sequência Y;
                iv) calcula as matrizes de sequência Z;
                v) calcula a matriz de sequência de forma completa;
                vi) calcula as correntes de falta, tensões pós falta e correntes pós falta do sistema em sequência;
                vii) calcula as correntes de falta, tensões pós falta e correntes pós falta do sistema em fase;
                viii) declara o DataFrame;
                ix) ajusta seus índices;
                x) imprime o DataFrame no terminal;
                xi) finaliza a função retornando o DataFrame;

    Entrada(s):

    Saída(s):

    Referência(s):
    """

    def nomesLinhas(Zbarra):

        """
        Descrição: subfunção que apenas ajusta o índice das linhas do DataFrame.
        """

        nomes = [None]*len(Zbarra)
        for index, linha in enumerate(nomes, 0):
            if index%3 == 0: fase = 'A'
            elif index%3 == 1: fase = 'B'
            elif index%3 == 2: fase = 'C'
            nomes[index] = f"Barra {index//3+1} fase {fase}"
        return nomes

    I012fsistema, V012posfsistema, I012posfsistema = sistemaPosfalta(Z012, p, tipoFalta, Za, Zb, Zc, Zg)
    Iabcfsistema, Vabcposfsistema, Iabcposfsistema = fortescue(I012fsistema), fortescue(V012posfsistema), fortescue(I012posfsistema)
    db = pd.DataFrame({'Ifalta': Iabcfsistema, 'Vpósfalta': Vabcposfsistema, 'Ipósfalta': Iabcposfsistema})
    db.index = nomesLinhas(Z012)
    return db


# Questão 7.1
def intervencao(linha, Y):

    """
    Descrição: função que recalcula Y com base na linha a ser retirada. Para tal, basta considerar 
                que a admitância da linha vai para 0.

    Entrada(s):

    Saída(s):
    """

    barraDe, barraPara = dictlinha(L2)['De'][linha-1], dictlinha(L2)['Para'][linha-1]
    Y[barraDe-1][barraDe-1] = 0
    Y[barraDe-1][barraPara-1] = 0
    Y[barraPara-1][barraDe-1] = 0
    Y[barraPara-1][barraPara-1] = 0
    return Y


def casosFalta(L2, Sbase, Vbase, Za, Zb, Zc, Zg, LX):

    """
    Descrição: função que realiza todos os cenários de falta possíveis. Para tal,

                i) determina variável dbs (lista de DataFrames com dados dos cenários);
                ii) calcula conforme padrões requeridos, os dados do sistema;
                iii) determina o Zbase;
                iv) calcula as matrizes de admitância nodal;
                v) calcula as matrizes de impedância nodal;
                vi) calcula a matriz de sequência;
                vii) itera-se sobre as barras:
                    vii.a) para cada barra, itera-se sobre os casos de curto-circuito:
                        vii.a.1) calcula o comportamento do circuito especificado o tipo de curto-circuito;
                        vii.a.2) adiciona os comportamentos à variável dbs;
                viii) finaliza a função retornando dbs;

    Entrada(s):

    Saída(s):
    """

    dbs = list()
    dictBarras, dictLinhas, dictTrafos, dictGers = parametrosANAREDE(L2, Sbase, Vbase, dictbarra, dictlinha, dicttrafo, dictger)
    Zbase = Vbase**2/Sbase
    Y0, Y1, Y2 = list(map(lambda seq: yBarra(dictLinhas, dictTrafos, dictGers, seq, Zbase), list(range(3))))
    print(f'Intervenção na Linha: {LX}\n')
    Z0, Z1, Z2 = list(map(lambda Y: linalg.inv(Y) if LX == 0 else linalg.inv(intervencao(LX, Y)), [Y0, Y1, Y2]))
    Z012 = matrizZ012(Z0, Z1, Z2)
    barras, casos = list(range(1, len(Z012)//3 + 1)), ['trifasica', 'faseterra', 'fasefase', 'fasefaseterra']
    for barra in barras:
        for caso in casos:
            db = baseDados(Z012, barra, caso, Za, Zb, Zc, Zg)
            dbs.append(db)
            print(f"\n\nFalta {caso} na barra {barra}\n{db}")
    return dbs


L2, Sbase, Vbase, Za, Zb, Zc, Zg = 47, 100, 230, 1, 1, 5, 100
#casosFalta(L2, Sbase, Vbase, Za, Zb, Zc, Zg)


# Questão 8.2 Intervenções de LTs
casosFalta(L2, Sbase, Vbase, Za, Zb, Zc, Zg, 0)


def intervencaoLT():

    """
    Descrição: calcula todas as intervenções possíveis caso não especificado.
    """

    for linha in dictlinha['Num.']:
        casosFalta(L2, Sbase, Vbase, Za, Zb, Zc, Zg, linha)
    return
