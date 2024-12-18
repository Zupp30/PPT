from sympy import *
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# from Chuong2 import *
def intro_ch2():
    st.header(":material/home: Giới thiệu", divider="blue")
    st.subheader("1. Phương pháp chia đôi")
    st.markdown("Tham khảo tại [đây](https://timoday.edu.vn/wp-content/uploads/2020/03/Phuong-phap-chia-doi.pdf)")
    st.subheader("2. Phương pháp dây cung")
    st.markdown("Tham khảo tại [đây](https://timoday.edu.vn/wp-content/uploads/2020/03/Phuong-phap-day-cung.pdf)")
    st.divider()
def plot_ch2(f, a, b, xn=None):
    x = Symbol('x')
    x_vals = np.linspace(a, b, 100)
    x_vals = [float(x_val) for x_val in x_vals]
    y_vals = [f.subs(x, x_val) for x_val in x_vals]
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.grid()
    if xn:
        ax.scatter(xn, f.subs(x, xn), color="red", label="Nghiệm gần đúng")
    ax.legend()
    st.pyplot(fig)
def handle_error_ch2(f, a, b):
    x = Symbol('x')
    f1 = diff(f, x)
    for _ in range(10):
        r = np.random.uniform(0.001, 0.499)
        if f.subs(x, a) * f.subs(x, b) > 0 or f1.subs(x, a+r) * f1.subs(x, b-r) < 0:
            return False
    return True
def manage_input_ch2():
    x = Symbol('x')
    f = st.text_input("Nhập vào hàm số f(x): ")
    if f:
        f = sympify(f)
        st.write(f)
        col1, col2 = st.columns(2, border=True)
        with col1:
            left, right = st.select_slider(label="Xác định khoảng", options=np.arange(-10, 11, 1), value=(0, 1))
            left, right = float(left), float(right)
        with col2:
            x_temp = np.linspace(left, right, int(right-left) * 10000)
            y_temp = lambdify(x, f)(x_temp)
            roots_temp = []

            for i in range(len(y_temp)-1):
                ele_l = y_temp[i]
                ele_r = y_temp[i+1]
                if ele_l * ele_r <= 0:
                    roots_temp.append((x_temp[i], x_temp[i+1]))

            roots_temp = pd.DataFrame(roots_temp, columns=["a", "b"])
            st.caption("Bảng gợi ý các khoảng ly nghiệm hợp lệ")
            st.dataframe(roots_temp, use_container_width=True, hide_index=True)
        st.info("Nên chọn hai cận gần nhất với các giá trị a, b ở bảng trên")
        a = st.text_input("Nhập vào cận trái `a`: ", value=None)
        b = st.text_input("Nhập vào cận phải `b`: ", value=None)
        s = st.text_input("Nhập vào sai số `s`: ", value=None)
        if a and b and s:
            a, b, s = float(a), float(b), float(s)
            if a < b and handle_error_ch2(f, a, b):
                return f, a, b, s
            else:
                st.error("Khoảng ly nghiệm không hợp lệ hoặc a phải nhỏ hơn b")
    return None, None, None, None
def chia_doi_ch2(f, a, b, s):
    x = Symbol('x')
    a_vals, b_vals, c_vals = [a], [b], []
    while True:
        c = (a_vals[-1] + b_vals[-1]) / 2
        c_vals.append(c)
        ss = (b_vals[-1] - a_vals[-1]) / 2
        if f.subs(x, c) == 0 or abs(ss) < s:
            break
        elif f.subs(x, a_vals[-1]) * f.subs(x, c) < 0:
            a_vals.append(a_vals[-1])
            b_vals.append(c)
        else:
            a_vals.append(c)
            b_vals.append(b_vals[-1])
    df = pd.DataFrame({"a": a_vals, "b": b_vals, "c": c_vals, "f(c)": [round(f.subs(x, i), 8) for i in c_vals]})
    st.markdown(f"Nghiệm gần đúng của phương trình là:")
    st.latex(f"{latex(c_vals[-1])} \plusmn {latex(ss)}")
    col1, col2 = st.columns(2)
    with col1.expander("Bảng giá trị"):
        st.dataframe(df, use_container_width=True)
    with col2.expander("Biểu đồ hàm số"):
        plot_ch2(f, a_vals[0], b_vals[0], c_vals[-1])
    return True
def day_cung_ch2(f, a, b, s):
    x = Symbol('x')
    a0, b0 = a, b
    c_vals, ss_vals = [], []
    f1 = diff(f, x)
    f2 = diff(f1, x)
    for _ in range(10):
        r = np.random.uniform(0.001, 0.499)
        if f2.subs(x, a+r) * f2.subs(x, b-r) < 0:
            st.write("Khoảng ly nghiệm không hợp lệ")
            return False
    M = max(f1.subs(x, a), f1.subs(x, b))
    m = min(f1.subs(x, a), f1.subs(x, b))
    while True:
        if f.subs(x, b) * f2.subs(x, b+r) < 0:
            c = b - (f.subs(x, b) * (b - a)) / (f.subs(x, b) - f.subs(x, a))
            ss = (M - m) * abs(c - b) / m
            b = c
        else:
            c = a - (f.subs(x, a) * (b - a)) / (f.subs(x, b) - f.subs(x, a))
            ss = (M - m) * abs(c - a) / m
            a = c
        if abs(ss) < s:
            break
        c_vals.append(c)
        ss_vals.append(ss)
    df = pd.DataFrame({"c": c_vals, "ss": ss_vals})
    st.markdown(f"Nghiệm gần đúng của phương trình là:")
    st.latex(f"{latex(c_vals[-1])} \plusmn {latex(ss_vals[-1])}")
    col1, col2 = st.columns(2)
    with col1.expander("Bảng giá trị"):
        st.dataframe(df, use_container_width=True)
    with col2.expander("Biểu đồ hàm số"):
        plot_ch2(f, a0, b0, c_vals[-1])
    return True
def tim_nghiem_ch2():
    st.header("Phương pháp chia đôi và dây cung", divider="blue")
    f, a, b, s = manage_input_ch2()
    if f:
        choice = st.selectbox("Chọn phương pháp", ["Chia đôi", "Dây cung"])
        if choice == "Chia đôi":
            chia_doi_ch2(f, a, b, s)
        elif choice == "Dây cung":
            day_cung_ch2(f, a, b, s)

# from Chuong4 import *
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

# from Chuong5 import *
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

# from Chuong6 import *
def intro_ch6():
    st.header(":material/home: Giới thiệu", divider="blue")
    st.write("Phương trình vi phân thường bậc nhất có dạng:")
    st.latex(r"""
                \begin{align*}
                y' = f(x, y) \\
                y(a) = y_0
                \end{align*}
            """)
    st.write("Trong đó:")
    st.write("- `f(x, y)` là hàm số đã biết")
    st.write("- `a` là điểm bắt đầu")
    st.write("- `b` là điểm kết thúc")
    st.write("- `h` là bước nhảy")
    st.write("- `y(a)` là giá trị ban đầu")
    st.write("Các phương pháp giải gần đúng phương trình vi phân thường bậc nhất:")
    st.write("1. Phương pháp Euler")
    st.write("2. Phương pháp Euler cải tiến")
    st.divider()
def manage_input_ch6():
    x_arr, y_arr, ff, f_a, a, h = None, None, None, None, None, None
    col1, col2 = st.columns(2)
    with col1:
        x, y = symbols('x y')
        ff = st.text_input("Nhập hàm `y'`:", placeholder="y'")
        if ff:
            st.latex("y' = " + ff)
            ff = lambdify([x, y], ff)
        
    with col2:
        col21, col22 = st.columns(2)
        with col21:
            a_temp = st.text_input("Nhập `a`:", placeholder="a", value=None)
            b_temp = st.text_input("Nhập `b`:", placeholder="b", value=None)
        with col22:
            h_temp = st.text_input("Nhập `h`:", placeholder="h", value=None)
            f_a_temp = st.text_input("Nhập `y(a)`:", placeholder="y(a)", value=None)
        if a_temp and b_temp and h_temp and f_a_temp:
            a, b, h = float(a_temp), float(b_temp), float(h_temp)
            f_a = float(f_a_temp)
            x_arr = np.arange(a, b+h, h)
            y_arr = [f_a]
    return x_arr, y_arr, ff, f_a, a, h
def Euler_ch6(x_arr, y_arr, ff, f_a, a, h):
    st.write("Phương pháp Euler:")
    st.latex("y_{i+1} = y_i + h*f(x_i, y_i)")
    f_arr = [ff(a, f_a)]
    hf_arr = [h*ff(a, f_a)]
    for i in range(1, len(x_arr)):
        t1 = y_arr[i-1]
        t2 = h*ff(x_arr[i-1], y_arr[i-1])
        if i != len(x_arr)-1:
            f_arr.append(t1)
            hf_arr.append(t2)
        else:
            f_arr.append(None)
            hf_arr.append(None)
        y_arr.append(t1 + t2)
    if x_arr is not None:
        df = pd.DataFrame({
            'x': x_arr,
            'y': y_arr,
            'f(x, y)': f_arr,
            'h*f(x, y)': hf_arr
        })
    st.dataframe(df, use_container_width=True, key='Euler')
def Heun_ch6(x_arr, y_arr, ff, f_a, a, h):
    st.write("Phương pháp Euler cải tiến:")
    st.latex(r"""
             \begin{align*}
             y_{i+1} &= y_i + \frac{h}{2} \left( f(x_i, y_i) + f(x_{i+1}, y_i + h*f(x_i, y_i)) \right) \\
            \end{align*}""")
    f_arr = [ff(a, f_a)]
    hf_arr = [h*ff(a, f_a)]
    for i in range(1, len(x_arr)):
        t1 = y_arr[i-1]
        t2 = h/2 * (ff(x_arr[i-1], y_arr[i-1]) + ff(x_arr[i], y_arr[i-1] + h*ff(x_arr[i-1], y_arr[i-1])))
        if i != len(x_arr)-1:
            f_arr.append(t1)
            hf_arr.append(t2)
        else:
            f_arr.append(None)
            hf_arr.append(None)
        y_arr.append(t1 + t2)
    if x_arr is not None:
        df = pd.DataFrame({
            'x': x_arr,
            'y': y_arr,
            'f(x, y)': f_arr,
            'h*f(x, y)': hf_arr
        })
    st.dataframe(df, use_container_width=True, key='Heun')
def tim_nghiem_ch6():
    st.header("Tìm nghiệm", divider="blue")
    x_arr, y_arr, ff, f_a, a, h = manage_input_ch6()
    if x_arr is not None and y_arr is not None:
        choice = st.selectbox("Chọn phương pháp giải:", ["Euler", "Euler cải tiến"])
        if choice == "Euler":
            Euler_ch6(x_arr, y_arr, ff, f_a, a, h)
        elif choice == "Euler cải tiến":
            Heun_ch6(x_arr, y_arr, ff, f_a, a, h)
        # elif choice == "Euler cải tiến hiệu chỉnh dần":
        #     proHeun_ch6(x_arr, y_arr, ff, f_a, a, h)


def main_intro():
    st.title("Phương pháp tính")
    st.markdown("Chào mừng đến với web app Phương pháp tính. Ứng dụng này hỗ trợ giải các bài toán về phương trình đại số, đa thức nội suy, đạo hàm và tích phân xác định, cũng như phương trình vi phân thường bậc nhất.")
    st.markdown("App được viết bởi sinh viên `Nguyễn Đình Huy Dũng - AT200214 - Lớp AT20B`")

    st.markdown("Chọn tính năng ở thanh bên trái")
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs")
    if selected == 1:
        st.markdown("Cảm ơn bạn đã thích, chúc bạn học tốt! :heart:")
    elif selected == 0:
        st.markdown("Rất tiếc vì bạn không thích, mình sẽ cố gắng hơn :broken_heart:")
    st.markdown("Hãy góp ý để ứng dụng ngày càng hoàn thiện hơn nhé! :smile:")

if __name__ == "__main__":
    st.set_page_config(page_title="Phương pháp tính", page_icon=":rocket:", layout="wide", initial_sidebar_state="collapsed")
    pages = {
        "TRANG CHỦ": [
            st.Page(main_intro, title="Lời giới thiệu", icon=":material/waving_hand:"),
        ],
        "CHƯƠNG 2: Giải gần đúng phương trình đại số": [
            st.Page(intro_ch2, title="Giới thiệu"),
            st.Page(tim_nghiem_ch2, title="Tìm nghiệm"),
        ],
        "CHƯƠNG 3: IN PROGRESS": [],
        "CHƯƠNG 4: Đa thức nội suy": [
            st.Page(intro_ch4, title="Giới thiệu"),
            st.Page(Horner_ch4, title="Sơ đồ Horner"),
            st.Page(Interpolate_ch4, title="Đa thức nội suy"),
        ],
        "CHƯƠNG 5: Tính gần đúng đạo hàm và tích phân xác định": [
            st.Page(intro_ch5, title="Giới thiệu"),
            st.Page(dao_ham_ch5, title="Tính gần đúng đạo hàm"),
            st.Page(tich_phan_ch5, title="Tính gần đúng tích phân xác định"),
        ],
        "CHƯƠNG 6: Giải gần đúng phương trình vi phân thường bậc nhất": [
            st.Page(intro_ch6, title="Giới thiệu"),
            st.Page(tim_nghiem_ch6, title="Tìm nghiệm"),
        ],
    }
    pg = st.navigation(pages)
    pg.run()
