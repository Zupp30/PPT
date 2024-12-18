import numpy as np
import pandas as pd
from sympy import *
import streamlit as st

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

# def proHeun_ch6(x_arr, y_arr, ff, f_a, a, h, tol=1e-4):
#     st.write("Phương pháp Euler cải tiến hiệu chỉnh dần:")
#     f_arr = [ff(a, f_a)]
#     hf_arr = [h*ff(a, f_a)]
#     for i in range(1, len(x_arr)):
#         t1 = y_arr[i-1]
#         t2 = h/2 * (ff(x_arr[i-1], y_arr[i-1]) + ff(x_arr[i], y_arr[i-1] + h*ff(x_arr[i-1], y_arr[i-1])))
        
    # if x_arr is not None:
    #     df = pd.DataFrame({
    #         'x': x_arr,
    #         'y': y_arr,
    #         'f(x, y)': f_arr,
    #         'h*f(x, y)': hf_arr
    #     })
    # st.dataframe(df, use_container_width=True, key='pro`Heun')

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

if __name__ == "__main__":
    st.set_page_config(page_icon=":1234:", page_title="Chương 6")
    st.title("Chương 6: Phương pháp giải gần đúng phương trình vi phân thường bậc nhất")
    pg = st.navigation([
        st.Page(page=intro_ch6, title="Giới thiệu", icon=":material/home:"),
        st.Page(page=tim_nghiem_ch6, title="Tìm nghiệm"),
    ])
    pg.run()
