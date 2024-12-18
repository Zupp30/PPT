from sympy import *
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

if __name__ == "__page__":
    st.set_page_config(layout="centered", page_icon=":1234:", page_title="Chương 2")
    st.title("Chương 2: Giải gần đúng phương trình đại số")
    pg = st.navigation([
        st.Page(page=intro_ch2, title="Giới thiệu", icon=":material/home:"),
        st.Page(page=tim_nghiem_ch2, title="Tìm nghiệm")
    ])
    pg.run()