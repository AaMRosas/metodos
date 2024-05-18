# libraries
import streamlit as st
import numpy as np
import pandas as pd
import math
from sympy import *
import sympy as sym
from fractions import Fraction
import re
import time
import requests  
from PIL import Image
from io import BytesIO
import sympy
import matplotlib.pyplot as plt
import cmath
x = symbols("x")

def ddn(m, z):
    # Crear matriz para diferencias divididas
    a = []
    for g in range(len(m) + 1):
        aux = [0] * len(m)
        a.append(aux)

    for s in range(len(m)):
        a[0][s] = m[s]
        a[1][s] = z[s]

    b = 1
    c = 1
    d = 1
    w = 0
    for i in range(len(a[0])):
        for j in range(len(a[0]) - b):
            a[c + 1][j] = (a[c][j + 1] - a[c][j]) / (a[0][j + d] - a[0][j])
        b += 1
        c += 1
        d += 1

    # Transponer y redondear la matriz para visualización
    matrix = np.transpose(a)
    matrix_r = np.round(matrix, decimals=4)
    return matrix_r

# Función para calcular la matriz Jacobiana
def matrizJacobiano(variables, funciones):
    n = len(funciones)
    m = len(variables)
    # Inicializar la matriz Jacobiana con ceros
    Jcb = sym.zeros(n, m)
    for i in range(n):
        unafi = sym.sympify(funciones[i])
        for j in range(m):
            unavariable = variables[j]
            Jcb[i, j] = sym.diff(unafi, unavariable)
    return Jcb

# Configuración básica
st.set_page_config(
    page_title="Mi Aplicación",
    page_icon="📊",
    layout="wide"
)

# Estilo para cambiar el color de la barra lateral
sidebar_style = """
<style>
.css-1v3fvcr {
    background-color: #8B4513; /* Marrón oscuro */
    color: #F5DEB3; /* Trigo */
}
</style>
"""

# Aplicar el estilo a la barra lateral
st.markdown(sidebar_style, unsafe_allow_html=True)

# Placeholder para la imagen
image_placeholder = st.empty()
# URL de la imagen
image_url = "https://raw.githubusercontent.com/AaMRosas/metodos/main/1626144278711.jpg"

# Descargar la imagen y verificar el resultado
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
  
with image_placeholder.container():
    st.image(image, width=300, caption="Ayuda")

# Simular espera
time.sleep(2)

# Limpiar el placeholder
image_placeholder.empty()

# Barra lateral con opciones
with st.sidebar:
    st.header("Métodos Numéricos II")

    num_method = st.radio(
        "Elija uno de los siguientes métodos numéricos",
        [
            "Introduccion",
            "Newton-Raphson",
            "Punto Fijo",
            "Lagrange",
            "Diferencias Divididas",
            "Mínimos Cuadrados",
            "Método del Trapecio",
            "Método de Simpson 1/3",
            "Método de Simpson 3/8"
        ]
    )
    

if num_method == "Introduccion":
    st.markdown("<h1 style='text-align: center;'>Universidad Nacional Autónoma de México</h1>", unsafe_allow_html=True)
    st.write("\t ## Matemáticas Aplicadas y Computación")
    # URLs de las imágenes en formato raw de GitHub
    image_url1 = "https://raw.githubusercontent.com/AaMRosas/metodos/main/mac.png"
    image_url2 = "https://raw.githubusercontent.com/AaMRosas/metodos/main/UNAM.png"
    
    # Definir las columnas con diferentes anchos
    col_uno, col_dos = st.columns([1, 2], gap="small")
    
    # Contenido de la primera columna
    with col_uno:
        # Agregar una imagen en la primera columna desde una URL
        st.image(image_url1)
        
        # Agregar el texto en la primera columna
        st.write("## Métodos Numéricos II ")
    
    # Contenido de la segunda columna
    with col_dos:
        # Agregar una imagen en la segunda columna desde una URL
        st.image(image_url2, width=170, caption=None)
        
        # Agregar el texto en la segunda columna
        st.info(
            "### Integrantes:\n"
            "- Munive Rosas Arturo Alberto\n"
            "- Erick Mercado Alejandre\n"
            "- Vilchis López Víctor Manuel\n"
            "- Alexis Salgado Urtes\n"
            "- Enríquez Sánchez Joshua Antonio",
            icon="ℹ️"
        )






# Newton-Raphson
if num_method == "Newton-Raphson":

   def matrizJacobiano(variables, funciones):
        n = len(funciones)
        m = len(variables)
        # matriz Jacobiano inicia con ceros
        Jcb = sym.zeros(n, m)
        for i in range(0, n, 1):
            unafi = sym.sympify(funciones[i])
            for j in range(0, m, 1):
                unavariable = variables[j]
                Jcb[i, j] = sym.diff(unafi, unavariable)
        return Jcb

    # Definir título centralizado
    st.markdown("<h1 style='text-align: center;'>Newton-Raphson Multivariable</h1>", unsafe_allow_html=True)

    # Entrada de datos de las ecuaciones y punto inicial
    with st.form(key="my_form"):
        st.write("Ingrese las ecuaciones:")
        num_ecuaciones = st.number_input("Número de ecuaciones:", min_value=1, step=1, value=2)
        variables = [sym.Symbol(f'x{i}') for i in range(num_ecuaciones)]
        funciones = [st.text_input(f"Ecuación {i+1}:", key=f'func_{i}') for i in range(num_ecuaciones)]

        x0 = st.number_input("Valor inicial para x:", value=1.5)
        y0 = st.number_input("Valor inicial para y:", value=3.5)

        tolerancia = st.number_input("Tolerancia:", value=0.0001)

        submit_button = st.form_submit_button(label="Calcular")

    if submit_button:
        # Convertir las entradas en expresiones simbólicas
        funciones = [sym.sympify(func) for func in funciones]

        # Ajustar las variables y funciones si hay menos de 3 ecuaciones
        if num_ecuaciones < 2:
            variables.extend([sym.Symbol(f'x{i}') for i in range(num_ecuaciones, 2)])
            funciones.extend(['0'] * (2 - num_ecuaciones))

        # Calcular el Jacobiano
        Jxy = matrizJacobiano(variables, funciones)

        # Valores iniciales
        valores_iniciales = {variables[0]: x0, variables[1]: y0}

        # Inicializar iteraciones y tramo
        iteraciones = 0
        tramo = tolerancia * 2

        # Mostrar tabla con los resultados
        st.write("Tabla de Resultados:")
        st.write("| Iteración | Jacobiano | Determinante | (x, y) | Error |")
        st.write("|-----------|-----------|--------------|--------|-------|")

        while tramo > tolerancia:
            # Sustituir valores en el Jacobiano
            J = Jxy.subs(valores_iniciales)

            # Calculamos el determinante de J
            Jn = np.array(J, dtype=float)
            determinante = np.linalg.det(Jn)

            # Calculamos las funciones evaluadas en los puntos iniciales
            funciones_evaluadas = [func.subs(valores_iniciales) for func in funciones]

            # Calculamos los nuevos valores
            nuevos_valores = {}
            for i, var in enumerate(variables):
                numerador = sum(func_eval * Jn[i, j] for j, func_eval in enumerate(funciones_evaluadas))
                nuevos_valores[var] = valores_iniciales[var] - numerador / determinante

            # Calculamos el tramo
            tramo = max(abs(nuevos_valores[var] - valores_iniciales[var]) for var in variables)

            # Actualizamos valores iniciales
            valores_iniciales = nuevos_valores.copy()

            # Incrementamos el contador de iteraciones
            iteraciones += 1

            # Mostramos los resultados en la tabla
            st.write(f"| {iteraciones} | {J} | {determinante:.4f} | ({', '.join(f'{val:.4f}' for val in valores_iniciales.values())}) | {tramo:.6f} |")

        # Mostramos el resultado final
        st.success(f"Resultado final: ({', '.join(f'{val:.4f}' for val in valores_iniciales.values())})")


if num_method == "Diferencias Divididas":
    st.markdown("<h1 style='text-align: center;'>Diferencias Divididas</h1>", unsafe_allow_html=True)
    # Texto informativo
    st.info("El método de Newton de las diferencias divididas nos permite calcular los coeficientes $c_j$ de la combinación lineal mediante la construcción de las llamadas diferencias divididas, que vienen definidas de forma recurrente.")
    
    # Presentar fórmulas utilizando st.latex
    st.latex("f[x_i] = f_i")
    
    st.latex("f[x_i, x_{i+1}, \ldots, x_{i+j}] = \\frac{f[x_{i+1}, \ldots, x_{i+j}] - f[x_i, x_{i+1}, \ldots, x_{i+j-1}]}{x_{i+j} - x_i}")
    
    st.info("Tenemos los siguientes casos particulares:")
    
    st.latex("f[x_0, x_1] = \\frac{f[x_1] - f[x_0]}{x_1 - x_0}")
    
    st.latex("f[x_0, x_1, x_2] = \\frac{f[x_1, x_2] - f[x_0, x_1]}{x_2 - x_0}")


    with st.form(key="divided_diff_form"):
        num_puntos = st.number_input("Ingrese el número de puntos", min_value=2, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            x_i = st.text_area("Ingrese los valores de $x_i$ separados por comas (,)", value="0, 1, 2")
        with col2:
            f_i = st.text_area("Ingrese los valores de $f_i$ separados por comas (,)", value="1, 2, 3")
        
        submit_button = st.form_submit_button(label="Calcular")

    if submit_button:
        try:
            # Obtener valores de los campos de texto
            m = [float(x) for x in x_i.split(",")]
            z = [float(x) for x in f_i.split(",")]

            if len(m) != num_puntos or len(z) != num_puntos:
                st.error(f"Error: Las listas deben tener exactamente {num_puntos} elementos.")
            else:
                st.info(f"Tabla de valores con {num_puntos} puntos:")
                tabla = pd.DataFrame({"xi": m, "fi": z})
                st.dataframe(tabla)
                
                # Calcular las diferencias divididas
                dd_matrix = ddn(m, z)
                st.write("Tabla de diferencias:")
                st.dataframe(dd_matrix)
                
                # Construcción del polinomio de diferencias divididas
                a = []
                for g in range(len(m) + 1):
                    aux = []
                    for e in range(len(m)):
                        aux.append(0)
                    a.append(aux)
            
                for s in range(len(m)):
                    a[0][s] = m[s]
                    a[1][s] = z[s]

                b = 1
                c = 1
                d = 1
                w = 0  # Inicializa la variable w
                for i in range(len(a[0])):
                    for j in range(len(a[0]) - b):
                        a[c + 1][j] = (a[c][j + 1] - a[c][j]) / (a[0][j + d] - a[0][j])
                    b += 1
                    c += 1
                    d += 1
                print("\n")
                matrix = np.array(a)
                matrix_t = np.transpose(matrix)
                matrix_r=np.round(matrix_t, decimals=4)
                matrix_df = pd.DataFrame(matrix_r)
                print("Tabla De Diferencias:")
                print(matrix_df)
                # Se obtiene todo el polinomio
                p = 0  # Define polinomio inicialmente
                for t in range(len(a[0])):
                   terminos = 1
                   for r in range(w):
                       terminos *= (x - a[0][r])
                   w += 1  # Actualiza w
                   p += a[t + 1][0] * terminos
                pol = simplify(p)                
                    # Obtener los coeficientes del polinomio
                coefficients = pol.as_poly().all_coeffs()
                   # Convertir los coeficientes a fracciones
                coefficients_as_fractions = [Rational(coef).limit_denominator() for coef in coefficients]
                   # Reconstruir el polinomio a partir de los coeficientes fraccionarios
                polynomial_terms = [f"{coef}*x^{i}" for i, coef in enumerate(coefficients_as_fractions[::-1])]
                   # Imprimir el polinomio con los términos separados
                polynomial_expression = " + ".join(polynomial_terms)   
                sexo = simplify(polynomial_expression)
                st.markdown(f"**Polinomio de Diferencias Divididas:** {pol}")
                st.write(f"{sexo}") 
                sexo = str(sexo)
                # Tu ecuación dinámica (como cadena de texto)
                # Convertimos "**" a "^" para potencias
                latex_expression = sexo.replace("**", "^")
                
                # Eliminamos el símbolo "*" ya que en LaTex, la multiplicación es implícita
                latex_expression = latex_expression.replace("*", "")
                
                # Usamos una expresión regular para transformar divisiones en formato LaTex
                # Esto convierte x/y a \frac{x}{y}
                latex_expression = re.sub(r"(\d+|\w)/(\d+)", r"\\frac{\1}{\2}", latex_expression)
                
                # Renderizamos en LaTex con Streamlit
                st.latex(latex_expression)

                
                                
                

        except Exception as e:
            st.error(f"Ocurrió un error al procesar los datos: {e}")
           
# Interpolación de Lagrange
if num_method == "Lagrange":
    st.markdown("<h1 style='text-align: center;'>Interpolación de Lagrange</h1>", unsafe_allow_html=True)
    st.info("Este método es el más explícito para probar existencia de solución ya que la construye.Sin embargo su utilidad se reduce a eso: a dar una respuesta formal y razonada, pues no es eficiente en términos de cálculo (requiere muchas operaciones y tiene limitaciones técnicas)", icon="ℹ️")
    st.info(" La fórmula de interpolación de Lagrange es:")
    
    # Fórmula de interpolación de Lagrange
    st.latex("P(x) = \\sum_{k=0}^{n} f_k \\cdot l_k(x)")
    
    # Definición de l_k(x)
    st.latex("l_k(x) = \\prod_{j=0, \\; j \\neq k}^{n} \\frac{x - x_j}{x_k - x_j}, \\; \\text{para } k = 0, \\ldots, n.")

    # Entrada de datos para los valores de 'x_i' y 'f_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista '  x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    entrada_y = st.text_input("Ingrese los elementos de la lista 'f_i' separados por comas(,):")
    fi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    # Verificar que las listas tengan la misma longitud y al menos un elemento
    if len(xi) == len(fi) and len(xi) > 0:
        # Crear el polinomio de Lagrange
        n = len(xi)
        x = sympy.Symbol("x")
        polinomio = 0
        divisorL = np.zeros(n, dtype=float)

        # Calcular el polinomio de Lagrange
        for i in range(0, n, 1):
            numerador = 1
            denominador = 1
            for j in range(0, n, 1):
                if j != i:
                    numerador *= x - xi[j]
                    denominador *= xi[i] - xi[j]
            terminoLi = numerador / denominador
            polinomio += terminoLi * fi[i]
            divisorL[i] = denominador

        # Simplificar el polinomio
        polisimple = polinomio.expand()

        # Para evaluación numérica
        px = sympy.lambdify(x, polisimple)

        # Crear datos para el gráfico
        muestras = 101
        a = np.min(xi)
        b = np.max(xi)
        pxi = np.linspace(a, b, muestras)
        pfi = px(pxi)

        # Mostrar resultados
        st.write("Divisores en L(i):", divisorL)
        st.write("Polinomio de Lagrange (expresión):", polinomio)
        st.write("Polinomio de Lagrange (simplificado):", polisimple)

        # Crear y mostrar el gráfico
        fig, ax = plt.subplots()
        ax.plot(xi, fi, 'o', label='Puntos')
        ax.plot(pxi, pfi, label='Polinomio')
        ax.legend()
        ax.set_xlabel("xi")
        ax.set_ylabel("fi")
        ax.set_title("Interpolación de Lagrange")

        buf = BytesIO()  # Crear un buffer para el gráfico
        plt.savefig(buf, format='png')  # Guardar el gráfico en memoria
        buf.seek(0)  # Ir al inicio del buffer
        st.image(buf, use_column_width=True)  # Mostrar la imagen en Streamlit

    else:
        st.warning("Por favor, ingrese listas válidas para 'x_i' y 'f_i', asegurándose de que tengan la misma longitud.")



if num_method == "Mínimos Cuadrados":
    # Título de la aplicación
    st.markdown("<h1 style='text-align: center;'>Minimos Cuadrados(Regresión Lineal)</h1>", unsafe_allow_html=True)

    # Entrada de datos para los valores de 'x_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista 'x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    # Entrada de datos para los valores de 'y_i'
    entrada_y = st.text_input("Ingrese los elementos de la lista 'y_i' separados por comas(,):")
    yi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    if len(xi) == len(yi) and len(xi) > 0:
        # Procedimiento
        xi = np.array(xi, dtype=float)
        yi = np.array(yi, dtype=float)
        n = len(xi)

        # Sumatorias y medias
        xm = np.mean(xi)
        ym = np.mean(yi)
        sx = np.sum(xi)
        sy = np.sum(yi)
        sxy = np.sum(xi * yi)
        sx2 = np.sum(xi ** 2)
        sy2 = np.sum(yi ** 2)

        # Coeficientes a0 y a1
        a1 = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
        a0 = ym - a1 * xm
        a0 = np.round(a0, 4)
        a1 = np.round(a1, 4)
        # Polinomio de grado 1
        x = sym.Symbol('x')
        f = a0 + a1 * x

        fx = sym.lambdify(x, f)
        fi = fx(xi)

        # Coeficiente de correlación
        numerador = n * sxy - sx * sy
        raiz1 = np.sqrt(n * sx2 - sx ** 2)
        raiz2 = np.sqrt(n * sy2 - sy ** 2)
        r = numerador / (raiz1 * raiz2)

        # Coeficiente de determinación
        r2 = r ** 2
        r2_porcentaje = np.round(r2 * 100, 4)
        
        # Redondear r y r2 a 4 decimales
        r = np.round(r, 4)
        r2 = np.round(r2, 4)

        # Mostrar resultados
        st.write("Función de regresión lineal: ", f)
        st.write("Coeficiente de correlación (r): ", r)
        st.write("Coeficiente de determinación (r²): ", r2)
        st.write(f"{r2_porcentaje}% de los datos están descritos en el modelo lineal")

        # Gráfica
        fig, ax = plt.subplots()
        ax.plot(xi, yi, 'o', label='(xi, yi)')
        ax.plot(xi, fi, color='orange', label=f)

        # Líneas de error
        for i in range(0, n, 1):
            y0 = np.min([yi[i], fi[i]])
            y1 = np.max([yi[i], fi[i]])
            ax.vlines(xi[i], y0, y1, color='red', linestyle='dotted')
        ax.legend()
        ax.set_xlabel('xi')
        ax.set_title('Mínimos Cuadrados')

        # Convertir el gráfico a formato de imagen
        buf = BytesIO()
        plt.savefig(buf, format='png')
        st.image(buf, use_column_width=True)

    else:
        st.warning("Ingrese valores válidos para 'x_i' y 'y_i', y asegúrese de que ambas listas tengan la misma longitud.")

if num_method=="Punto Fijo":
    st.markdown("<h1 style='text-align: center;'>Método del Punto Fijo</h1>", unsafe_allow_html=True)

    # Entrada de datos para el valor inicial y el número máximo de iteraciones
    x_0 = st.number_input('Ingresa valor inicial (X_0):', format="%.4f")
    N = st.number_input('Número máximo de iteraciones:', min_value=1, step=1, format="%d")
    f = st.text_input('Ingrese la función g(x):')

    if st.button('Calcular'):
        if not f:
            st.warning("Por favor, ingrese una función válida para g(x).")
        else:
            tolerancia = 0.0001
            er = 100
            i = 1
            x = x_0

            st.write('Iteración\tg(f(x))')
            st.write(f"0\t\t{x:.4f}")
            resultados = [(0, x)]

            while i <= N and er >= tolerancia:
                temp = x
                try:
                    x = eval(f, {'__builtins__': None}, {'x': temp, 'cmath': cmath, 'math': math, 'np': np})
                except Exception as e:
                    st.error(f"Error en la evaluación de la función: {e}")
                    break
                er = abs((x - temp))
                st.write(f"{i}\t\t{x:.4f}")
                resultados.append((i, x))
                i += 1

            st.write(f"\nLa solución más aproximada es: {x:.4f}")

            # Mostrar resultados en una tabla
            st.table(resultados)

            # Gráfica
            iteraciones, valores = zip(*resultados)
            fig, ax = plt.subplots()
            ax.plot(iteraciones, valores, marker='o', linestyle='-', color='b', label='g(f(x))')
            ax.set_xlabel('Iteraciones')
            ax.set_ylabel('Valor de g(f(x))')
            ax.set_title('Convergencia del Método del Punto Fijo')
            ax.legend()

            # Convertir el gráfico a formato de imagen
            buf = BytesIO()
            plt.savefig(buf, format='png')
            st.image(buf, use_column_width=True)


if num_method=="Método del Trapecio":
    def Trapecio(a, b, n, fi, xi):
        k = 0
        suma = 0
        trapezoids = []  # Lista para almacenar los puntos de los trapezoides
        iteraciones = []  # Lista para almacenar los datos de las iteraciones

        h = (b - a) / n
        if equid(xi):
            for i in range(k, n):
                area = (fi[i] + fi[i + 1]) / 2  # Área del trapecio
                suma += area
                trapezoids.append([(xi[i], 0), (xi[i], fi[i]), (xi[i + 1], fi[i + 1]), (xi[i + 1], 0)])
                iteraciones.append([i+1, xi[i], xi[i+1], fi[i], fi[i+1], area])

            resultado = h * suma
            return round(resultado, 3), trapezoids, iteraciones
        else:
            st.error("Valores no equidistantes")
            return None, None, None

# Función para verificar equidistancia
    def equid(xi):
        tol = 1e-10  # Tolerancia para verificar equidistancia
        diff = xi[1] - xi[0]  # Diferencia entre el segundo y primer elemento

        for i in range(1, len(xi) - 1):
            # Verificar si la diferencia entre los elementos es aproximadamente igual
            if abs(xi[i + 1] - xi[i] - diff) > tol:
                return False
        return True

    # Título de la aplicación
    st.markdown("<h1 style='text-align: center;'>Método del Trapecio</h1>", unsafe_allow_html=True)

    # Entrada de datos para los valores de 'x_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista 'x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    # Entrada de datos para los valores de 'f(x_i)'
    entrada_y = st.text_input("Ingrese los elementos de la lista 'f(x_i)' separados por comas(,):")
    fi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    if len(xi) == len(fi) and len(xi) > 1:
        n = len(xi) - 1
        A = xi[0]
        B = xi[-1]

        resultado_integral, trapezoids, iteraciones = Trapecio(A, B, n, fi, xi)

        if resultado_integral is not None:
            st.write("El resultado de la integral es:", resultado_integral)

            # Mostrar iteraciones en una tabla
            iteraciones_df = pd.DataFrame(iteraciones, columns=["Iteración", "x_i", "x_i+1", "f(x_i)", "f(x_i+1)", "Área"])
            st.table(iteraciones_df)

            # Gráfica de la función f(x)
            x = np.linspace(A, B, 100)
            y = np.interp(x, xi, fi)  # Interpolación lineal para obtener valores de y en x

            fig, ax = plt.subplots()
            ax.plot(x, y, label='f(x)')
            ax.scatter(xi, fi, color='red', label='Puntos')

            # Dibujar los trapezoides utilizados para la integración
            for trap in trapezoids:
                trap_x = [p[0] for p in trap]
                trap_y = [p[1] for p in trap]
                ax.fill(trap_x, trap_y, color='green', alpha=0.3)

            # Etiquetas y leyenda
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Método del Trapecio')
            ax.legend()

            # Convertir el gráfico a formato de imagen
            buf = BytesIO()
            plt.savefig(buf, format='png')
            st.image(buf, use_column_width=True)

    else:
        st.warning("Ingrese valores válidos para 'x_i' y 'f(x_i)', y asegúrese de que ambas listas tengan la misma longitud y más de un punto.")


if num_method=="Método de Simpson 1/3":
    def integrasimpson13_fi(xi, fi, tolera=0.001):
        n = len(xi)
        i = 0
        suma = 0
        iteraciones = []  # Lista para almacenar los datos de las iteraciones

        while not (i >= (n - 2)):
            h = xi[i + 1] - xi[i]
            dh = abs(h - (xi[i + 2] - xi[i + 1]))
            if dh < tolera:  # tramos iguales
                unS13 = (h / 3) * (fi[i] + 4 * fi[i + 1] + fi[i + 2])
                suma += unS13
                iteraciones.append([xi[i], xi[i + 1], xi[i + 2], fi[i], fi[i + 1], fi[i + 2], unS13])
                i += 2
            else:  # tramos desiguales
                st.error("Los intervalos no son equidistantes")
                return None, None
        if i < (n - 1):  # incompleto, faltan tramos por calcular
            st.error("Faltan puntos")
            return None, None
        return round(suma, 3), iteraciones

# Título de la aplicación
    st.markdown("<h1 style='text-align: center;'>Método de Simpson 1/3</h1>", unsafe_allow_html=True)

    # Entrada de datos para los valores de 'x_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista 'x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    # Entrada de datos para los valores de 'f(x_i)'
    entrada_y = st.text_input("Ingrese los elementos de la lista 'f(x_i)' separados por comas(,):")
    fi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    if len(xi) == len(fi) and len(xi) > 2:
        resultado_integral, iteraciones = integrasimpson13_fi(xi, fi)

        if resultado_integral is not None:
            st.write("El resultado de la integral es:", resultado_integral)

            # Mostrar iteraciones en una tabla
            iteraciones_df = pd.DataFrame(iteraciones, columns=["x_i", "x_i+1", "x_i+2", "f(x_i)", "f(x_i+1)", "f(x_i+2)", "Área"])
            st.table(iteraciones_df)

            # Gráfica de la función f(x)
            x = np.linspace(min(xi), max(xi), 100)
            y = np.interp(x, xi, fi)  # Interpolación lineal para obtener valores de y en x
            plt.plot(x, y, label='f(x)')

            # Graficar los puntos
            plt.scatter(xi, fi, color='red', label='Puntos')

            # Etiquetas y leyenda
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Gráfico de f(x)')
            plt.legend()

            # Dibujar los tramos utilizados para la integración
            for i in range(len(xi) - 2):
                plt.plot(xi[i:i+3], fi[i:i+3], color='green')

            st.pyplot(plt)

    else:
        st.warning("Ingrese al menos 3 valores válidos para 'x_i' y 'f(x_i)', y asegúrese de que ambas listas tengan la misma longitud.")

if num_method=="Método de Simpson 3/8":
    def integrasimpson38_fi(xi, fi, tolera=0.001):
        n = len(xi)
        i = 0
        suma = 0
        iteraciones = []
        while i <= (n - 4):
            h = xi[i+1] - xi[i]
            h1 = xi[i+2] - xi[i+1]
            h2 = xi[i+3] - xi[i+2]
            dh = abs(h - h1) + abs(h - h2)
            if dh < tolera:  # tramos iguales
                unS38 = fi[i] + 3*fi[i+1] + 3*fi[i+2] + fi[i+3]
                unS38 = (3/8) * h * unS38
                suma += unS38
                iteraciones.append([xi[i], xi[i+1], xi[i+2], xi[i+3], fi[i], fi[i+1], fi[i+2], fi[i+3], unS38])
                i += 3  # avanzar 3 intervalos
            else:  # tramos desiguales
                print("\nLos intervalos no son equidistantes")
                return None, None
        if (i+3) < n:  # incompleto, tramos por calcular
            print("\nFaltan puntos")
            return None, None
        return round(suma, 3), iteraciones

# Título de la aplicación
    st.markdown("<h1 style='text-align: center;'>Método de Simpson 3/8</h1>", unsafe_allow_html=True)

    # Entrada de datos para los valores de 'x_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista 'x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    # Entrada de datos para los valores de 'f(x_i)'
    entrada_y = st.text_input("Ingrese los elementos de la lista 'f(x_i)' separados por comas(,):")
    fi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    if len(xi) == len(fi) and len(xi) > 3:
        resultado_integral, iteraciones = integrasimpson38_fi(xi, fi)

        if resultado_integral is not None:
            st.write("El resultado de la integral es:", resultado_integral)

            # Mostrar iteraciones en una tabla
            iteraciones_df = pd.DataFrame(iteraciones, columns=["x_i", "x_i+1", "x_i+2", "x_i+3", "f(x_i)", "f(x_i+1)", "f(x_i+2)", "f(x_i+3)", "Área"])
            st.table(iteraciones_df)

            # Gráfica de la función f(x)
            x = np.linspace(min(xi), max(xi), 100)
            y = np.interp(x, xi, fi)  # Interpolación lineal para obtener valores de y en x
            plt.plot(x, y, label='f(x)')

            # Graficar los puntos
            plt.scatter(xi, fi, color='red', label='Puntos')

            # Etiquetas y leyenda
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Gráfico de f(x)')
            plt.legend()

            # Dibujar los tramos utilizados para la integración
            for i in range(0, len(xi) - 3, 3):
                plt.plot(xi[i:i + 4], fi[i:i + 4], color='green')

            st.pyplot(plt)

    else:
        st.warning("Ingrese al menos 4 valores válidos para 'x_i' y 'f(x_i)', y asegúrese de que ambas listas tengan la misma longitud.")