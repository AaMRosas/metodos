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
        for i in range(n):
            unafi = sym.sympify(funciones[i])
            for j in range(m):
                unavariable = variables[j]
                Jcb[i, j] = sym.diff(unafi, unavariable)
        return Jcb

    # Título de la aplicación
    st.markdown("<h1 style='text-align: center;'>Método de Newton-Raphson para Sistemas No Lineales</h1>", unsafe_allow_html=True)

    # Entrada de datos para las funciones y valores iniciales
    f1 = st.text_input("Ingrese la primera función f1(x, y):")
    f2 = st.text_input("Ingrese la segunda función f2(x, y):")

    x0 = st.number_input("Ingrese el valor inicial para x0:", format="%.4f")
    y0 = st.number_input("Ingrese el valor inicial para y0:", format="%.4f")

    tolera = st.number_input("Ingrese la tolerancia:", value=0.0001, format="%.4f")

    if st.button('Calcular'):
        if not f1 or not f2:
            st.warning("Por favor, ingrese funciones válidas.")
        else:
            # Definir símbolos
            x = sym.Symbol('x')
            y = sym.Symbol('y')
            
            # Convertir cadenas a funciones simbólicas
            f1 = sym.sympify(f1)
            f2 = sym.sympify(f2)

            # Procedimiento
            funciones = [f1, f2]
            variables = [x, y]

            Jxy = matrizJacobiano(variables, funciones)

            # Valores iniciales
            xi = x0
            yi = y0

            # Tramo inicial, mayor que tolerancia
            itera = 0
            tramo = tolera * 2

            resultados = []
            st.write("### Iteraciones")
            while tramo > tolera:
                J = Jxy.subs([(x, xi), (y, yi)])

                # Determinante de J
                Jn = np.array(J, dtype=float)
                determinante = round(np.linalg.det(Jn),4)

                # Iteraciones
                f1i = f1.subs([(x, xi), (y, yi)])
                f2i = f2.subs([(x, xi), (y, yi)])

                numerador1 = f1i * Jn[1, 1] - f2i * Jn[0, 1]
                xi1 = xi - numerador1 / determinante
                numerador2 = f2i * Jn[0, 0] - f1i * Jn[1, 0]
                yi1 = yi - numerador2 / determinante

                tramo = np.max(np.abs([xi1 - xi, yi1 - yi]))
                xi = round(xi1, 4)  # Redondear a 4 decimales
                yi = round(yi1, 4)  # Redondear a 4 decimales

                itera += 1
                resultados.append((itera, J, determinante, xi, yi, tramo))

                st.write(f"**Iteración {itera}**")
                st.write("Jacobiano con puntos iniciales:")
                st.write(J)
                st.write(f"Determinante: {determinante}")
                st.write(f"Puntos xi, yi: {xi}, {yi}")
                st.write(f"Error: {tramo}")

            st.write("### Resultado final")
            st.write(f"x: {xi}, y: {yi}")

            # Mostrar resultados en una tabla
            st.write("### Detalle de las iteraciones")
            iteracion_data = {
                "Iteración": [res[0] for res in resultados],
                "Jacobian": [str(res[1]) for res in resultados],
                "Determinante": [res[2] for res in resultados],
                "xi": [res[3] for res in resultados],
                "yi": [res[4] for res in resultados],
                "Error": [res[5] for res in resultados],
            }
            st.table(iteracion_data)




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

if num_method == "Punto Fijo":

    # Funciones de iteración
    def g1(x, y):
        return (x**2 + y**2 + 8) / 10

    def g2(x, y):
        return (x*y**2 + x + 8) / 10

    # Método de Jacobi
    def jacobi_method(x0, y0, tol, nmaxiter):
        error = 100
        niter = 0
        results = []
        results.append((niter, x0, y0, error))

        while error > tol and niter < nmaxiter:
            x1 = g1(y0, x0)
            y1 = g2(x0, y0)
            niter += 1
            error = max(abs(x1 - x0), abs(y1 - y0))
            results.append((niter, x1, y1, error))
            x0 = x1
            y0 = y1
        return results

    # Método de Gauss-Seidel
    def gauss_seidel_method(x0, y0, tol, nmaxiter):
        error = 100
        niter = 0
        results = []
        results.append((niter, x0, y0, error))

        while error > tol and niter < nmaxiter:
            x1 = g1(x0, y0)
            y1 = g2(x1, y0)
            niter += 1
            error = max(abs(x1 - x0), abs(y1 - y0))
            results.append((niter, x1, y1, error))
            x0 = x1
            y0 = y1
        return results

    # Aplicación Streamlit
    st.title("Punto fijo")
    st.info("Suponemos que se quiere buscar una solución del sistema:")

    with st.container():
        st.latex("x^2 - 10x + y^2 + 8 = 0,")
        st.latex("xy^2 + x - 10y + 8 = 0.")

    st.info("Se eligen las siguientes ecuaciones para el método del punto fijo:")

    with st.container():
        st.latex("x = \\frac{1}{10}(x^2 + y^2 + 8),")
        st.latex("y = \\frac{1}{10}(xy^2 + x + 8),")
        
    st.info("Se partirá de los valores que den para $(x,y)$.")

    x0 = st.number_input("Ingrese el valor inicial para x:", value=0.0, format="%.5f")
    y0 = st.number_input("Ingrese el valor inicial para y:", value=0.0, format="%.5f")
    tol = st.number_input("Ingrese la tolerancia:", value=0.00001, format="%.5f")
    nmaxiter = st.number_input("Ingrese el número máximo de iteraciones:", value=100, step=1)

    if st.button("Calcular"):
        st.subheader("Resultados del Método de Jacobi")
        jacobi_results = jacobi_method(x0, y0, tol, nmaxiter)
        jacobi_df = pd.DataFrame(jacobi_results, columns=["Iteración", "x", "y", "Error"])
        st.dataframe(jacobi_df.style.format({"x": "{:.5f}", "y": "{:.5f}", "Error": "{:.5f}"}))

        st.subheader("Resultados del Método de Gauss-Seidel")
        gauss_seidel_results = gauss_seidel_method(x0, y0, tol, nmaxiter)
        gauss_seidel_df = pd.DataFrame(gauss_seidel_results, columns=["Iteración", "x", "y", "Error"])
        st.dataframe(gauss_seidel_df.style.format({"x": "{:.5f}", "y": "{:.5f}", "Error": "{:.5f}"}))

            