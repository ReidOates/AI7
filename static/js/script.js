document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    // UI Elements
    const welcomeMessage = document.getElementById('welcomeMessage');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('resultContent');
    const riskBadge = document.getElementById('riskBadge');
    const confidenceValue = document.getElementById('confidenceValue');

    // Show Loader
    welcomeMessage.style.display = 'none';
    resultContent.style.display = 'none';
    loader.style.display = 'block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (result.error) {
            alert('Error: ' + result.error);
            welcomeMessage.style.display = 'block';
            loader.style.display = 'none';
            return;
        }

        // Fill Results
        riskBadge.innerText = result.risk === 'Tinggi' ? 'RISIKO TINGGI' : 'RISIKO RENDAH';
        riskBadge.className = 'risk-badge ' + (result.risk === 'Tinggi' ? 'risk-high' : 'risk-low');
        confidenceValue.innerText = result.confidence;
        
        if (result.risk === 'Tinggi') {
            confidenceValue.classList.remove('text-success');
            confidenceValue.classList.add('text-danger');
        } else {
            confidenceValue.classList.remove('text-danger');
            confidenceValue.classList.add('text-success');
        }

        // Hide loader and show result
        setTimeout(() => {
            loader.style.display = 'none';
            resultContent.style.display = 'block';
        }, 800);

    } catch (error) {
        console.error('Error:', error);
        alert('Terjadi kesalahan saat menghubungi server.');
        welcomeMessage.style.display = 'block';
        loader.style.display = 'none';
    }
});
