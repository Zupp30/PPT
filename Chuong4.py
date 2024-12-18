import numpy as np
import pandas as pd
from sympy import *
import streamlit as st
import matplotlib.pyplot as plt

def Horner_ch4():
    st.header("1. Sơ đồ Horner", divider="blue")
    x = Symbol("x")
    deg = int(st.number_input("Nhập bậc của đa thức `P(x)`", min_value=1, value=1))
    coeffs = []
    cols = st.columns(deg + 1, border=True)
    for i in range(deg + 1):
        with cols[i]:
            num = st.text_input(f"Nhập hệ số `x^{deg-i}`", value=None)
            if num:
                coeffs.append(Number(num))
    if len(coeffs) == deg + 1:
        P = sum([coeffs[i] * x**(deg-i) for i in range(deg + 1)])
        st.markdown("Đa thức `P(x)` là:")
        st.write(P)
        c = st.number_input("Nhập giá trị `c`:", value=0)
        st.markdown("Tính giá trị của `P(x)` tại `x = {}` bằng sơ đồ Horner".format(c))
        add = [0]
        result = []
        for i in range(deg):
            result.append(coeffs[i] + c * (result[i - 1] if i > 0 else 0))
            add.append(result[i] * c)
        result.append(coeffs[-1] + c * result[-1])
        df = pd.DataFrame({
            "Hệ số": coeffs,
            "Cộng": add,
            "Kết quả": result
        })
        df = df.T
        df.columns = ["a{}".format(i) for i in range(deg + 1)]
        st.dataframe(df, use_container_width=True)

        st.markdown("Kết quả: `P({}) = {}`".format(c, result[-1]))
        st.divider()

def plot_ch4(f, x_values, y_values, name, x_val, y_val):
    x = Symbol("x")
    f = lambdify(x, f, "numpy")
    x_range = np.linspace(min(x_values), max(x_values), 100)
    y_range = f(x_range)
    fig = plt.figure()
    plt.plot(x_range, y_range, label="Đa thức nội suy " + name)
    plt.scatter(x_values, y_values, color="red", label="Điểm nội suy")
    if x_val is not None and y_val is not None:
        plt.scatter(x_val, y_val, color="green", label="Giá trị tại điểm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    st.pyplot(fig)

def calc_ch4(name, f):
    with st.popover("Tính giá trị tại một điểm", use_container_width=True):
        x_temp = st.text_input("Nhập giá trị `x`:", value=None, key=f"{name}_x")
        if x_temp:
            x_val = Number(x_temp)
            x = Symbol("x")
            y_val = f.subs(x, x_val)
            st.write("Kết quả: `{}({}) = {}`".format(name, x_val, y_val))
            return x_val, y_val
    return None, None

def manage_values_ch4():
    st.header("2. Đa thức nội suy", divider="blue")
    n = st.number_input("Nhập số điểm nội suy `n`", min_value=2, value=2)
    x_values, y_values = [], []
    for i in range(n):
        col1, col2 = st.columns(2, border=True)
        with col1:
            x_temp = st.text_input(f"Nhập x{i}", value=0)
            if x_temp:
                x_value = Number(x_temp)
                x_values.append(x_value)
        with col2:
            y_temp = st.text_input(f"Nhập y{i}", value=0)
            if y_temp:
                y_value = Number(y_temp)
                y_values.append(y_value)
    return x_values, y_values, n

def Lagrange_ch4(x_values, y_values, n):
    x = Symbol("x")
    L = 0
    for i in range(n):
        Li = 1
        for j in range(n):
            if j != i:
                Li *= (x - x_values[j]) / (x_values[i] - x_values[j])
        L += Li * y_values[i]
    if L != nan:
        st.markdown("Theo công thức, đa thức nội suy Lagrange là:")
        st.write(L)
        L = simplify(L)
        st.markdown("Đa thức nội suy Lagrange sau khi rút gọn:")
        st.write(L)
        col1, col2 = st.columns(2, border=True)
        with col1.expander("Xem bảng giá trị"):
            df_values = pd.DataFrame({
                "x": x_values,
                "y": y_values
            })
            st.dataframe(df_values, use_container_width=True, hide_index=True)
            x_val, y_val = calc_ch4("L", L)
        with col2.expander("Xem biểu đồ"):
            plot_ch4(L, x_values, y_values, "Lagrange", x_val, y_val)

    st.divider()

def Newton_ch4(x_values, y_values, n):
    x = Symbol("x")
    D = [[None] * n for _ in range(n)]
    D = np.array(D)
    D[:, 0] = y_values
    for i in range(1, n):
        for j in range(n - i):
            D[j, i] = (D[j + 1, i - 1] - D[j, i - 1]) / (x_values[j + i] - x_values[j])
    N = D[0, 0]
    for i in range(1, n):
        temp = 1
        for j in range(i):
            temp *= (x - x_values[j])
        N += D[0, i] * temp
    if N != nan:
        print(D)
        D = pd.DataFrame(D)
        D.columns = ["SP_{}".format(i) for i in range(n)]
        D.insert(0, "x", x_values)
        st.markdown("Bảng sai phân:")
        if st.checkbox("Hiện bảng sai phân"):
            st.dataframe(D, use_container_width=True, hide_index=True)
        st.markdown("Theo công thức, đa thức nội suy Newton tiến là:")
        st.write(N)
        N = simplify(N)
        st.markdown("Đa thức nội suy Newton sau khi rút gọn:")
        st.write(N)
        col1, col2 = st.columns(2, border=True)
        with col1.expander("Xem bảng giá trị"):
            df_values = pd.DataFrame({
                "x": x_values,
                "y": y_values
            })
            st.dataframe(df_values, use_container_width=True, hide_index=True)
            x_val, y_val = calc_ch4("N", N)

        with col2.expander("Xem biểu đồ"):
            plot_ch4(N, x_values, y_values, "Newton", x_val, y_val)
    st.divider()

def Interpolate_ch4():
    x_values, y_values, n = manage_values_ch4()
    if x_values != [0, 0] != y_values != [0, 0]:
        tab1, tab2 = st.tabs(["Lagrange", "Newton"])
        with tab1:
            Lagrange_ch4(x_values, y_values, n)
        with tab2:
            Newton_ch4(x_values, y_values, n)
    else:
        st.warning("Dữ liệu không hợp lệ, vui lòng sửa lại")

def intro_ch4():
    st.header(":material/home: Giới thiệu", divider="blue")
    st.subheader("1. Sơ đồ Horner")
    st.markdown("Sơ đồ Horner là một phương pháp giúp tính giá trị của một đa thức `P(x)` tại một giá trị `c` nào đó. "
                "Phương pháp này giúp giảm số lượng phép tính so với cách truyền thống. "
                "Sơ đồ Horner thường được sử dụng trong việc chia đa thức cho một đa thức tuyến tính `x - c`.")
    st.subheader("2. Đa thức nội suy")
    st.markdown("Đa thức nội suy là một phương pháp giúp xây dựng một đa thức `P(x)` đi qua `n` điểm đã biết. "
                "Đa thức nội suy có thể giúp xấp xỉ giá trị của một hàm tại các điểm không biết. "
                "Có hai phương pháp chính để xây dựng đa thức nội suy mà chúng ta sẽ tìm hiểu trong bài này là `Lagrange` và `Newton`.")
    st.divider()
# Main
if __name__ == "__page__":
    st.set_page_config(layout="wide", page_icon=":bar_chart:", page_title="Chương 4")
    st.title("Chương 4: Đa thức nội suy")
    pg = st.navigation([
        st.Page(page=intro_ch4, title="Giới thiệu", icon=":material/home:"),
        st.Page(page=Horner_ch4, title="Sơ đồ Horner"),
        st.Page(page=Interpolate_ch4, title="Đa thức nội suy")
    ])
    pg.run()
