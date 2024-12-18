from sympy import *
import numpy as np
import pandas as pd
import streamlit as st

def intro_ch5():
    st.header(":material/home: Giới thiệu", divider="blue")
    st.subheader("1. Tính gần đúng đạo hàm")
    st.subheader("a. Trường hợp hai nút nội suy ")
    st.write("Để tính gần đúng đạo hàm của hàm số f(x) tại x = x0 và x = x1, ta sử dụng công thức nút nội suy hai nút:")
    st.latex(r"f'(x_0) = \frac{f(x_1) - f(x_{0})}{h} - \frac{h}{2}f''(c_0), c_0  \in [x_0, x_1]")
    st.latex(r"f'(x_1) = \frac{f(x_1) - f(x_{0})}{h} + \frac{h}{2}f''(c_1), c_1  \in [x_0, x_1]")

    st.subheader("b. Trường hợp ba nút nội suy")
    st.write("Để tính gần đúng đạo hàm của hàm số f(x) tại x = x0, x = x1 và x = x2, ta sử dụng công thức nút nội suy ba nút:")
    st.latex(r"f'(x_0) = \frac{1}{2h}(-3f(x_0) + 4f(x_1) - f(x_2)) + \frac{h^2}{3}f'''(c_0), c_0  \in [x_0, x_2]")
    st.latex(r"f'(x_1) = \frac{1}{2h}(-f(x_0) + f(x_2)) - \frac{h^2}{6}f'''(c_1), c_1  \in [x_0, x_2]")
    st.latex(r"f'(x_2) = \frac{1}{2h}(f(x_0) - 4f(x_1) + 3f(x_2)) + \frac{h^2}{3}f'''(c_2), c_2  \in [x_0, x_2]")

    st.divider()

    st.subheader("2. Tính gần đúng tích phân xác định")
    st.subheader("a. Công thức hình thang và sai số")
    st.write("Để tính gần đúng tích phân xác định của hàm số f(x) trên đoạn [a, b], ta sử dụng công thức hình thang:")
    st.latex(r"\int_{a}^{b} f(x)dx \approx  \frac{h}{2}\sum_{i=0}^{n-1} (y_{i} + y_{i+1})")
    st.latex(r"h = \frac{b - a}{n}")
    st.write("Với sai số:")
    st.latex(r"s = \frac{(b - a)h^2}{12}f''(c), c \in [a, b]")

    st.subheader("b. Công thức Simpson tổng quát và sai số")
    st.write("Để tính gần đúng tích phân xác định của hàm số f(x) trên đoạn [a, b], ta sử dụng công thức nút nội suy Simpson:")
    st.latex(r"\int_{a}^{b} f(x)dx \approx  \frac{h}{3}\sum_{i=0}^{2m-2} (y_{i} + 4y_{i+1} + y_{i+2})")
    st.latex(r"h = \frac{b - a}{2m}")
    st.write("Với sai số:")
    st.latex(r"s = \frac{(b - a)h^4}{180}f^{(4)}(c), c \in [a, b]")

    st.divider()

def dao_ham_ch5():
    st.header("1. Tính gần đúng đạo hàm", divider="blue")
    k = st.number_input("Nhập số nút nội suy:", min_value=2, max_value=3, value=2)
    cols = st.columns(k, border=True)
    x_values = []
    y_values = []
    for i in range(k):
        with cols[i]:
            x = st.text_input(f"Nhập x{i}:", value=None)
            y = st.text_input(f"Nhập y{i}:", value=None)
            if x is not None and y is not None:
                x, y = float(x), float(y)
                x_values.append(x)
                y_values.append(y)

    if x_values != [] and y_values != []:
        df = pd.DataFrame({
            "x": x_values,
            "y": y_values
        })
        df = df.T
        df.columns = ["x{}".format(i) for i in range(k)]
        st.dataframe(df, use_container_width=True, hide_index=True)

        if k == 2:
            h = x_values[1] - x_values[0]
            f = (y_values[1] - y_values[0]) / h
            st.latex(r"f'(x_0) \approx \frac{f(x_1) - f(x_{0})}{h}")
            st.markdown("Đạo hàm của hàm số tại x0 là: `f'(x0) = {}`".format(f))
            st.latex(r"f'(x_1) \approx \frac{f(x_1) - f(x_{0})}{h}")
            st.markdown("Đạo hàm của hàm số tại x1 là: `f'(x1) = {}`".format(f))
        else:

            h = x_values[1] - x_values[0]
            f0 = (1 / (2 * h)) * (-3 * y_values[0] + 4 * y_values[1] - y_values[2])
            f1 = (1 / (2 * h)) * (-y_values[0] + y_values[2])
            f2 = (1 / (2 * h)) * (y_values[0] - 4 * y_values[1] + 3 * y_values[2])
            st.latex(r"f'(x_0) \approx \frac{1}{2h}(-3f(x_0) + 4f(x_1) - f(x_2))")
            st.markdown("Đạo hàm của hàm số tại x0 là: `f'(x0) = {}`".format(f0))
            st.latex(r"f'(x_1) \approx \frac{1}{2h}(-f(x_0) + f(x_2))")
            st.markdown("Đạo hàm của hàm số tại x1 là: `f'(x1) = {}`".format(f1))
            st.latex(r"f'(x_2) \approx \frac{1}{2h}(f(x_0) - 4f(x_1) + 3f(x_2))")
            st.markdown("Đạo hàm của hàm số tại x2 là: `f'(x2) = {}`".format(f2))

def manage_input_ch5():
    st.header("2. Tính gần đúng tích phân xác định", divider="blue")
    f = st.text_input("Nhập hàm số f(x):")
    if f:
        x = Symbol("x")
        f = simplify(f)
        st.write(f)
        a = st.text_input("Nhập cận dưới:")
        b = st.text_input("Nhập cận trên:")
        if a and b:
            return f, float(a), float(b)
    return None, None, None

def hinh_thang_ch5(f, a, b):
    n = st.number_input("Nhập số đoạn chia:", min_value=1)
    x = Symbol("x")
    xs = np.linspace(a, b, n + 1)
    xs = [float(xi) for xi in xs]
    ys = [f.subs(x, xi) for xi in xs]
    h = (b - a) / n
    res = 0
    for i in range(n):
        res += ys[i] + ys[i + 1]
    res *= h / 2
    s = max([diff(f, x, 2).subs(x, xi) for xi in xs]) * (b - a) * h ** 2 / 12
    st.latex(r"\int_{a}^{b} f(x)dx \approx  \frac{h}{2}\sum_{i=0}^{n-1} (y_{i} + y_{i+1})")
    st.markdown("Kết quả: `I = {}`".format(res))
    st.latex(r"s = \frac{(b - a)h^2}{12}f''(c), c \in [a, b]")
    st.markdown("Sai số: `s = {}`".format(s))
    if st.checkbox("Hiện bảng giá trị", key="hinh_thang"):
        df = pd.DataFrame({
            "x": xs,
            "y": ys
        })
        st.dataframe(df, use_container_width=True)

def simpson_ch5(f, a, b):
    n = st.number_input("Nhập số đoạn chia:", min_value=2, step=2)
    if n%2 == 1:
        st.error("Số đoạn chia phải là số chẵn")
        return
    x = Symbol("x")
    xs = np.linspace(a, b, n + 1)
    xs = [float(xi) for xi in xs]
    ys = [f.subs(x, xi) for xi in xs]
    h = (b - a) / n
    res = 0
    _s1 = _s2 = 0
    for i in range(2, n+2, 2):
        res += ys[i-2] + 4 * ys[i-1] + ys[i]
        if i< n:
            _s1 += ys[i]
        _s2 += ys[i-1]
    res *= h / 3
    s = max([diff(f, x, 4).subs(x, xi) for xi in xs]) * (b - a) * h ** 4 / 180
    st.latex(r"\int_{a}^{b} f(x)dx \approx  \frac{h}{3}\sum_{i=0}^{2m-2} (y_{i} + 4y_{i+1} + y_{i+2})")
    st.markdown("Kết quả: `I = {}`".format(res))
    st.latex(r"s = \frac{(b - a)h^4}{180}f^{(4)}(c), c \in [a, b]")
    st.markdown("Sai số: `s = {}`".format(s))
    if st.checkbox("Hiện bảng giá trị", key="simpson"):
        col1, col2 = st.columns(2)
        with col1:
            df = pd.DataFrame({
                "x": xs,
                "y": ys,
            })
            st.dataframe(df, use_container_width=True)
        with col2:
            df = pd.DataFrame({
                "y_le": [None] + ys[1::2],
                "y_chan": ys[::2],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("Tổng lẻ: `{}`".format(_s1))
            st.markdown("Tổng chẵn: `{}`".format(_s2))

def tich_phan_ch5():
    f, a, b = manage_input_ch5()
    if f is not None and a is not None and b is not None:
        tab1, tab2 = st.tabs(["Công thức hình thang", "Công thức Simpson tổng quát"])
        with tab1:
            hinh_thang_ch5(f, a, b)
        with tab2:
            simpson_ch5(f, a, b)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_icon="writing_hand", page_title="Chương 5")
    st.title("Chương 5: Tính gần đúng đạo hàm và tích phân xác định")
    pg = st.navigation([
        st.Page(page=intro_ch5, title="Giới thiệu", icon=":material/home:"),
        st.Page(page=dao_ham_ch5, title="Tính gần đúng đạo hàm"),
        st.Page(page=tich_phan_ch5, title="Tính gần đúng tích phân xác định")
    ])
    pg.run()