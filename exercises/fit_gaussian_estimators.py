from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    samples = np.random.normal(10, 1, 1000)
    u_gauss = UnivariateGaussian(biased_var=False)
    u_gauss.fit(samples)
    print("Question 1 - Draw samples and print fitted model")
    print(round(u_gauss.mu_,3),", ",round(u_gauss.var_,3))

    # Question 2 - Empirically showing sample mean is consistent

    distances = []
    size = []
    estimated_means=[]
    for n in range(10, 1001, 10):
        u_gauss.fit(samples[:n])
        estimated_means.append(u_gauss.mu_)
        distances.append(np.abs(u_gauss.mu_-10))
        size.append(n)
    graph = go.Figure()
    graph.add_trace(go.Scatter(x=size,y=distances,mode='lines+markers',
                               marker=dict(size=8, color='blue'),name='Absolute Distance'))
    graph.update_layout(
        xaxis_title='Sample Size',
        yaxis_title='Absolute Distance',
        title='Absolute Distance Between Estimated - and True Mean\n as a Function of Sample Size'
    )
    graph.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    u_gauss.fit(samples)
    pdf_values = u_gauss.pdf(samples)
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_pdf_values = pdf_values[sorted_indices]
    graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=sorted_samples, y=sorted_pdf_values,mode='markers',
                               marker=dict(size=5, color='blue'),name='Absolute Distance') )
    graph2.update_layout(
        xaxis_title='Values of samples',
        yaxis_title='PDF',
        title='PDF Function with fitted model'
    )
    graph2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    U = np.array([0, 0, 4, 0])
    SIGMA = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(U, SIGMA, 1000)
    multivariate = MultivariateGaussian()
    multivariate.fit(samples)
    print("multivariate mu")
    print(np.round(multivariate.mu_,3))
    print("multivariate cov")
    print(np.round(multivariate.cov_,3))
    # Question 5 - Likelihood evaluation

    f1 = f3 = np.linspace(-10, 10, 200)
    LL_values = np.zeros((f1.size, f3.size))
    for i in range(f1.size):
        for j in range(f3.size):
            new_mean = np.array([f1[i], 0, f3[j], 0])
            LL_values[i][j] = multivariate.log_likelihood(new_mean, SIGMA, samples)

    fig = go.Figure(data=go.Heatmap(
        z=LL_values,
        x=f3,
        y=f1,
        colorscale='Viridis'))

    fig.update_layout(
        title='Log Likelihood Heatmap',
        xaxis_title='f1 values',
        yaxis_title='f3 values')
    fig.show()
    # Question 6 - Maximum likelihood
    argmax_ll = np.argmax(LL_values)
    x,y = np.unravel_index(argmax_ll,LL_values.shape)
    print(np.around(f1[x], 3), np.around(f3[y], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
