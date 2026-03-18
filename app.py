import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.optimize import minimize
from fpdf import FPDF
import scipy.stats as stats
import sys
import types
import os
from itertools import combinations

# --- АВТОМАТИЧНО СПРАВЯНЕ С ГРЕШКАТА ЗА pyDOE ---
try:
    from pyDOE3 import bbdesign
except ImportError:
    try:
        if 'imp' not in sys.modules:
            sys.modules['imp'] = types.ModuleType('imp')
        from pyDOE2 import bbdesign
    except ImportError:
        st.error("🚨 Липсва pyDOE3. Напишете в терминала: pip install pyDOE3")
        st.stop()

# --- ФУНКЦИЯ ЗА PDF ДОКЛАД НА БЪЛГАРСКИ ---
def create_pdf_report(summary_df, opt_conditions_real, max_yield, r_squared, eq_str, matrix_df, pareto_path, surface_paths):
    pdf = FPDF()
    pdf.add_page()
    
    # Добавяне на кирилски шрифт (Изисква arial.ttf в папката!)
    font_path = "arial.ttf"
    has_cyrillic = os.path.exists(font_path)
    if has_cyrillic:
        pdf.add_font('ArialCustom', '', font_path, uni=True)
        font_name = 'ArialCustom'
    else:
        font_name = 'Helvetica' # Резервен вариант, който не чете кирилица добре
    
    # Заглавие
    pdf.set_font(font_name, size=16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 10, "Доклад: Оптимизация на Екстракция (Box-Behnken)", ln=True, align='C')
    if not has_cyrillic:
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(190, 10, "WARNING: arial.ttf not found. Cyrillic text may not display correctly.", ln=True, align='C')
    pdf.ln(5)
    
    # R-squared и Уравнение
    pdf.set_font(font_name, size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 8, f"Точност на модела (R-squared): {r_squared:.4f}", ln=True)
    pdf.ln(2)
    pdf.set_font(font_name, size=10)
    pdf.multi_cell(190, 6, f"Регресионно уравнение (кодирани стойности):\n{eq_str}")
    pdf.ln(5)

    # --- ЕКСПЕРИМЕНТАЛНА МАТРИЦА ---
    pdf.set_font(font_name, size=12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 10, "1. Експериментална матрица (Реални стойности)", ln=True)
    
    pdf.set_font(font_name, size=8)
    pdf.set_text_color(0, 0, 0)
    col_widths = [190 / len(matrix_df.columns)] * len(matrix_df.columns)
    
    # Хедър на матрицата
    for i, col_name in enumerate(matrix_df.columns):
        pdf.cell(col_widths[i], 6, str(col_name)[:15], border=1, align='C')
    pdf.ln()
    
    # Данни от матрицата
    for _, row in matrix_df.iterrows():
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 6, f"{val:.2f}", border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    # --- ТАБЛИЦА ЗА ANOVA ---
    if pdf.get_y() > 220: pdf.add_page()
    pdf.set_font(font_name, size=12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 10, "2. Статистически анализ (ANOVA)", ln=True)
    
    pdf.set_font(font_name, size=10)
    pdf.set_fill_color(200, 220, 255)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(80, 8, "Параметър", border=1, fill=True, align='C')
    pdf.cell(35, 8, "Коефициент", border=1, fill=True, align='C')
    pdf.cell(35, 8, "p-value", border=1, fill=True, align='C')
    pdf.cell(40, 8, "Значимост", border=1, fill=True, align='C', ln=True)
    
    for _, row in summary_df.iterrows():
        status = "Значим" if row['p-value'] < 0.05 else "Незначим"
        pdf.cell(80, 8, str(row['Параметър']), border=1)
        pdf.cell(35, 8, f"{row['Коефициент']:.4f}", border=1, align='C')
        pdf.cell(35, 8, f"{row['p-value']:.4f}", border=1, align='C')
        if row['p-value'] < 0.05:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(40, 8, status, border=1, align='C', ln=True)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.cell(40, 8, status, border=1, align='C', ln=True)
    pdf.ln(5)
    
    # --- ПАРЕТО ДИАГРАМА ---
    if os.path.exists(pareto_path):
        if pdf.get_y() > 150: pdf.add_page()
        pdf.set_font(font_name, size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(190, 10, "3. Парето диаграма на ефектите", ln=True)
        pdf.image(pareto_path, x=15, w=170)
        pdf.ln(5)

    # --- ОПТИМАЛНИ УСЛОВИЯ ---
    if pdf.get_y() > 220: pdf.add_page()
    pdf.set_font(font_name, size=12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(190, 10, "4. Оптимални работни условия", ln=True)
    
    pdf.set_font(font_name, size=11)
    pdf.set_text_color(0, 0, 0)
    for name, val in opt_conditions_real.items():
        pdf.cell(190, 7, f"   > {name}: {val:.2f}", ln=True)
    
    pdf.ln(3)
    pdf.set_fill_color(220, 255, 220)
    pdf.cell(190, 10, f" ПРОГНОЗИРАН МАКСИМАЛЕН ДОБИВ: {max_yield:.3f}", ln=True, fill=True, align='C')
    pdf.ln(10)

    # --- ВСИЧКИ 3D ГРАФИКИ ---
    if surface_paths:
        pdf.add_page()
        pdf.set_font(font_name, size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(190, 10, "5. 3D Повърхности на отговора", ln=True)
        
        y_pos = pdf.get_y()
        for idx, path in enumerate(surface_paths):
            if os.path.exists(path):
                # Слагаме по 2 графики на страница
                if idx % 2 == 0 and idx != 0:
                    pdf.add_page()
                    y_pos = pdf.get_y()
                pdf.image(path, x=15, y=y_pos, w=160)
                y_pos += 120 # Отместване надолу за следващата снимка

    return bytes(pdf.output())

# --- КОНФИГУРАЦИЯ НА ПРИЛОЖЕНИЕТО ---
st.set_page_config(page_title="BBD Оптимизатор", layout="wide")
st.title("🌿 Box-Behnken Пълен Оптимизатор (Реални стойности)")

# --- СТЪПКА 1: НАСТРОЙКИ В SIDEBAR ---
st.sidebar.header("⚙️ Фактори и Граници")
num_factors = st.sidebar.number_input("Брой фактори:", min_value=3, max_value=10, value=3)

factors_config = []
factors_names = []

for i in range(num_factors):
    st.sidebar.markdown(f"**Фактор {i+1}**")
    f_name = st.sidebar.text_input(f"Име:", value=f"Фактор {i+1}", key=f"name_{i}")
    col1, col2 = st.sidebar.columns(2)
    f_min = col1.number_input(f"Мин (-1):", value=10.0, key=f"min_{i}")
    f_max = col2.number_input(f"Макс (+1):", value=50.0, key=f"max_{i}")
    
    factors_names.append(f_name)
    factors_config.append({'name': f_name, 'min': f_min, 'max': f_max})

st.sidebar.divider()

# Функция за преобразуване от кодирани към реални
def coded_to_real(coded_val, f_min, f_max):
    center = (f_max + f_min) / 2
    half_range = (f_max - f_min) / 2
    return center + coded_val * half_range

@st.cache_data
def generate_real_matrix(n, config):
    c_pts = {3:3, 4:3, 5:6}.get(n, 6)
    coded_matrix = bbdesign(n, center=c_pts)
    
    real_matrix = np.zeros_like(coded_matrix)
    for i in range(n):
        real_matrix[:, i] = coded_to_real(coded_matrix[:, i], config[i]['min'], config[i]['max'])
        
    df = pd.DataFrame(real_matrix, columns=[c['name'] for c in config])
    df['Добив (Отговор)'] = 0.0 
    return df, coded_matrix

# --- СТЪПКА 2: МАТРИЦА ---
st.header("1. Експериментална матрица (Реални стойности)")
df_real_init, raw_coded_matrix = generate_real_matrix(num_factors, factors_config)
edited_df = st.data_editor(df_real_init, use_container_width=True, num_rows="dynamic")
edited_df['Добив (Отговор)'] = pd.to_numeric(edited_df['Добив (Отговор)'], errors='coerce').fillna(0.0)

# --- СТЪПКА 3: АНАЛИЗ ---
if st.button("🚀 Изчисли и Генерирай Пълен Отчет", type="primary"):
    st.divider()
    
    # Винаги правим математиката с КОДИРАНИТЕ стойности (-1, 0, 1), за да е коректен RSM анализът!
    X_coded = raw_coded_matrix
    Y = edited_df['Добив (Отговор)'].values

    if np.all(Y == 0):
        st.error("❌ Моля, въведете резултати в колоната 'Добив (Отговор)'!")
    elif len(Y) != len(X_coded):
        st.error("❌ Броят на редовете не съвпада с дизайна. Моля, не изтривайте редове от базовата матрица.")
    else:
        try:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_coded)
            X_sm = sm.add_constant(X_poly)
            model = sm.OLS(Y, X_sm).fit()

            feature_names = poly.get_feature_names_out(factors_names)
            summary_df = pd.DataFrame({
                "Параметър": feature_names,
                "Коефициент": model.params[1:],
                "p-value": model.pvalues[1:],
                "t-value": np.abs(model.tvalues[1:])
            })

            # 1. Уравнение
            st.subheader("📐 Регресионно уравнение")
            st.caption("Забележка: Уравнението е изчислено спрямо кодираните стойности (-1 до +1), както изисква стандартът за RSM.")
            eq_str = f"Y = {model.params[0]:.3f}"
            for name, coef in zip(feature_names, model.params[1:]):
                sign = "+" if coef >= 0 else "-"
                eq_str += f" {sign} {abs(coef):.3f}*[{name}]"
            st.code(eq_str, wrap_lines=True)

            # 2. Парето Диаграма
            st.subheader("📊 Парето Диаграма")
            df_resid = model.df_resid
            t_critical = stats.t.ppf(1 - 0.025, df_resid)
            pareto_df = summary_df.sort_values(by="t-value", ascending=True)
            colors = ['#FF4B4B' if t > t_critical else '#1F77B4' for t in pareto_df['t-value']]

            fig_pareto = go.Figure(go.Bar(x=pareto_df['t-value'], y=pareto_df['Параметър'], orientation='h', marker_color=colors))
            fig_pareto.add_vline(x=t_critical, line_dash="dash", line_color="red", annotation_text="Критична граница")
            fig_pareto.update_layout(title="Парето диаграма на стандартизираните ефекти", height=500)
            st.plotly_chart(fig_pareto, use_container_width=True)

            # 3. Оптимизация
            st.subheader("🎯 Оптимални Условия (Реални стойности)")
            def objective(x):
                return -model.predict(np.insert(poly.transform([x]), 0, 1))[0] 

            bounds = [(-1, 1) for _ in range(num_factors)]
            opt_res = minimize(objective, [0]*num_factors, bounds=bounds, method='L-BFGS-B')
            opt_coords_coded = opt_res.x
            max_val = -opt_res.fun

            # Превръщаме кодирания оптимум в РЕАЛНИ стойности за показване
            opt_real_dict = {}
            cols = st.columns(num_factors)
            for i in range(num_factors):
                real_val = coded_to_real(opt_coords_coded[i], factors_config[i]['min'], factors_config[i]['max'])
                opt_real_dict[factors_names[i]] = real_val
                cols[i].metric(factors_names[i], f"{real_val:.2f}")
            st.success(f"✨ Прогнозиран максимален добив: **{max_val:.3f}**")

            # 4. АВТОМАТИЧНО ГЕНЕРИРАНЕ НА ВСИЧКИ 3D ГРАФИКИ
            st.subheader("🖼️ Всички 3D Повърхности на отговора")
            st.write("Графиките показват взаимодействията по двойки, докато останалите фактори са фиксирани на оптималните им стойности.")
            
            combos = list(combinations(range(num_factors), 2))
            saved_3d_paths = []
            
            for idx, (ix, iy) in enumerate(combos):
                fact_x, fact_y = factors_names[ix], factors_names[iy]
                
                # Създаваме мрежа от КОДИРАНИ стойности за предвиждането
                g_size = 40
                xx_c, yy_c = np.meshgrid(np.linspace(-1, 1, g_size), np.linspace(-1, 1, g_size))
                
                plot_data_c = np.array([opt_coords_coded] * (g_size**2))
                plot_data_c[:, ix] = xx_c.ravel()
                plot_data_c[:, iy] = yy_c.ravel()
                
                zz = model.predict(sm.add_constant(poly.transform(plot_data_c), has_constant='add')).reshape(xx_c.shape)
                
                # Преобразуваме осите в РЕАЛНИ стойности за визуализацията
                xx_r = coded_to_real(xx_c, factors_config[ix]['min'], factors_config[ix]['max'])
                yy_r = coded_to_real(yy_c, factors_config[iy]['min'], factors_config[iy]['max'])
                
                fig_3d = go.Figure(data=[go.Surface(z=zz, x=xx_r, y=yy_r, colorscale='Portland')])
                fig_3d.update_layout(
                    title=f"Взаимодействие: {fact_x} и {fact_y}",
                    scene=dict(xaxis_title=fact_x, yaxis_title=fact_y, zaxis_title="Добив"), 
                    width=700, height=600, margin=dict(l=0, r=0, b=0, t=40)
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Запазваме снимката за PDF
                path_3d = f"temp_3d_{idx}.png"
                try:
                    fig_3d.write_image(path_3d, scale=2)
                    saved_3d_paths.append(path_3d)
                except:
                    pass

            # 5. ГЕНЕРИРАНЕ НА PDF
            st.divider()
            st.write("⏳ Генериране на PDF отчет...")
            
            pareto_path = "temp_pareto.png"
            try:
                fig_pareto.write_image(pareto_path, scale=2)
            except:
                pass
            
            pdf_bytes = create_pdf_report(summary_df, opt_real_dict, max_val, model.rsquared, eq_str, edited_df, pareto_path, saved_3d_paths)
            
            if os.path.exists("arial.ttf"):
                btn_label = "📥 Изтегли Пълен PDF Доклад (на Български)"
            else:
                btn_label = "⚠️ Изтегли PDF (ПРЕДУПРЕЖДЕНИЕ: arial.ttf липсва, кирилицата може да е счупена)"

            st.download_button(label=btn_label, data=pdf_bytes, file_name="Extraction_Report_Full.pdf", mime="application/pdf", type="primary")
            
            # Почистване на временни файлове
            if os.path.exists(pareto_path): os.remove(pareto_path)
            for path in saved_3d_paths:
                if os.path.exists(path): os.remove(path)

        except Exception as e:
            st.error(f"❌ Възникна грешка при изчисленията. Детайли: {e}")