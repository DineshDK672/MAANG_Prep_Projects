import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.power import TTestIndPower


def ab_test_sim(sample_size, param_a, param_b, std_dev, test_type='z'):
    np.random.seed(42)

    # Creating synthetic samples for A and B
    sample_a = np.random.normal(loc=param_a, scale=std_dev, size=sample_size)
    sample_b = np.random.normal(loc=param_b, scale=std_dev, size=sample_size)

    # Perfoming the tests
    if test_type == 't':
        stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    else:
        pooled_se = np.sqrt(2*(std_dev**2/sample_size))
        stat = (np.mean(sample_a) - np.mean(sample_b))/pooled_se
        p_value = 2 * stats.norm.sf(abs(stat))

    # Plotting the results
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample_a, nbinsx=20,
                  name='Group A', opacity=0.6))
    fig.add_trace(go.Histogram(x=sample_b, nbinsx=20,
                  name='Group B', opacity=0.6))
    fig.update_layout(barmode='overlay',
                      title='Sample Distributions - Group A vs Group B')

    return {'stat': stat, 'p_value': p_value, 'plot': fig}


def power_analysis(effect_size, alpha, power=None, sample_size=None):
    analysis = TTestIndPower()

    if sample_size == None:
        sample_size = analysis.solve_power(
            effect_size, power=power, alpha=alpha, alternative='two-sided')
    elif power is None:
        power = analysis.power(
            effect_size, nobs1=sample_size, alpha=alpha, alternative='two-sided')

    return effect_size, alpha, power, sample_size


def power_curve(effect_size, sample_size, alpha):
    analysis = TTestIndPower()
    samples = np.arange(5, sample_size, 5)
    powers = analysis.power(effect_size, nobs1=samples,
                            alpha=alpha, alternative='two-sided')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=powers,
                  mode='lines+markers', name='Power Curve'))
    fig.update_layout(title='Power Curve vs Sample Size',
                      xaxis_title='Sample Size per Group', yaxis_title='Power')

    return fig


# Streamlit Dashboard
st.title("Stats Inference Dashboard")

# Sidebar for section selection
section = st.sidebar.radio(
    "Choose Section", ["A/B Test Simulation", "Power Analysis", "Distribution Viz"])

if section == "A/B Test Simulation":
    st.header("A/B Test Simulation")

    # Input widgets
    sample_size = st.slider("Sample Size", 10, 1000, 100, step=10)
    test_type = st.selectbox("Select Test Type", ["z", "t"])

    param_a = st.number_input("Group A Mean", 0.0, 1.0, 0.1)
    param_b = st.number_input("Group B Mean", 0.0, 1.0, 0.15)
    std_dev = st.number_input("Standard Deviation", min_value=0.01, value=0.1)

    if st.button("Run A/B Test Simulation"):
        results = ab_test_sim(sample_size, param_a,
                              param_b, std_dev, test_type)
        st.write(f"Test Statistic: {results['stat']:.4f}")
        st.write(f"P-value: {results['p_value']:.6f}")
        st.plotly_chart(results['plot'], use_container_width=True)

elif section == "Power Analysis":
    st.header("Power Analysis")
    effect_size = st.number_input(
        "Effect Size (Cohen's d)", min_value=0.01, max_value=5.0, step=0.01, value=0.5)
    alpha = st.number_input("Significance Level (Alpha)",
                            min_value=0.001, max_value=0.1, step=0.001, value=0.05)
    power = st.number_input("Power (1 - Beta)", min_value=0.0,
                            max_value=0.99, step=0.01, value=0.8)
    sample_size = st.number_input(
        "Sample Size per Group (enter 0 if unknown)", min_value=0, step=1, value=0)

    if st.button("Calculate Missing Parameter"):
        es, a, p, ss = power_analysis(
            effect_size, alpha, power if power > 0 else None, sample_size if sample_size > 0 else None)
        st.write(f"Effect Size: {es:.3f}")
        st.write(f"Alpha: {a:.3f}")
        st.write(f"Power: {p:.3f}")
        st.write(f"Sample Size per Group: {ss:.1f}")

    st.header("Power Curve")
    sample = st.slider(
        "Select Sample size range for Power Curve", 20, 2000, 20, 20)
    fig = power_curve(effect_size, sample, alpha)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Power curve for the following parameters")
    st.write("Effect Size: ", effect_size)
    st.write("Alpha: ", alpha)

elif section == "Distribution Viz":
    st.header("Distribution Visualization")

    dist_options = ['Normal', 't-distribution', 'Chi-square']
    selected_dists = st.multiselect(
        "Select Distributions to Overlay", dist_options, default=['Normal'])

    x_min = st.number_input("X-axis minimum", value=-5.0)
    x_max = st.number_input("X-axis maximum", value=5.0)
    x = np.linspace(x_min, x_max, 500)

    fig = go.Figure()

    if 'Normal' in selected_dists:
        mu = st.number_input("Normal Mean", value=0.0)
        sigma = st.number_input(
            "Normal Std Dev", min_value=0.1, value=1.0)
        y_norm = stats.norm.pdf(x, mu, sigma)
        fig.add_trace(go.Scatter(x=x, y=y_norm, mode='lines', name='Normal'))

    if 't-distribution' in selected_dists:
        df = st.slider("t-distribution degrees of freedom", 1, 30, 10)
        y_t = stats.t.pdf(x, df)
        fig.add_trace(go.Scatter(
            x=x, y=y_t, mode='lines', name='t-distribution'))

    if 'Chi-square' in selected_dists:
        df_chi = st.slider("Chi-square degrees of freedom", 1, 30, 2)
        # Chi-square is defined for x>=0, trim x accordingly
        x_chi = x[x >= 0]
        y_chi = stats.chi2.pdf(x_chi, df_chi)
        fig.add_trace(go.Scatter(x=x_chi, y=y_chi,
                      mode='lines', name='Chi-square'))

    fig.update_layout(title="Probability Distribution Functions",
                      xaxis_title="x", yaxis_title="Density")

    st.plotly_chart(fig, use_container_width=True)
