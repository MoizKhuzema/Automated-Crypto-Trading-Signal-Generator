document.addEventListener('DOMContentLoaded', function() {
    // Get the value selected by the profit margin slider
    const profitMarginSlider = document.getElementById('profit-margin');
    const profitValue = document.getElementById('profit-value');
    profitValue.innerHTML = profitMarginSlider.value + '%';

    profitMarginSlider.addEventListener('input', function() {
        profitValue.innerHTML = this.value + '%';
    });

    // Get the value selected by the risk tolerance slider
    const riskToleranceSlider = document.getElementById('risk-tolerance');
    const riskValue = document.getElementById('risk-value');
    riskValue.innerHTML = riskToleranceSlider.value + '%';

    riskToleranceSlider.addEventListener('input', function() {
        riskValue.innerHTML = this.value + '%';
    });

    // Add any additional functionality here

    // Ensure the initial values are set
    updateSliderValues();
});

function updateSliderValues() {
    const profitMarginSlider = document.getElementById('profit-margin');
    const profitValue = document.getElementById('profit-value');
    profitValue.innerHTML = profitMarginSlider.value + '%';

    const riskToleranceSlider = document.getElementById('risk-tolerance');
    const riskValue = document.getElementById('risk-value');
    riskValue.innerHTML = riskToleranceSlider.value + '%';
}

window.onload = function() {
    // Ensure the initial values are set when the page loads
    updateSliderValues();
};
