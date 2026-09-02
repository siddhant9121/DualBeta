import sys
import requests
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QTextEdit
)

API_URL = "http://localhost:8000/beta"


class DAPMWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAPM Beta Screener")
        self.resize(450, 350)

        # --- Widgets ---
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker, e.g. AAPL")

        self.fetch_button = QPushButton("Fetch")
        self.fetch_button.clicked.connect(self.fetch_beta)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        # --- Layout ---
        input_row = QHBoxLayout()
        input_row.addWidget(self.ticker_input)
        input_row.addWidget(self.fetch_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_row)
        main_layout.addWidget(QLabel("Results:"))
        main_layout.addWidget(self.result_box)

        self.setLayout(main_layout)

    def fetch_beta(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            self.result_box.setText("Please enter a ticker.")
            return

        self.result_box.setText(f"Fetching {ticker}...")
        try:
            response = requests.get(f"{API_URL}/{ticker}", timeout=15)
            response.raise_for_status()
            data = response.json()

            formatted = (
                f"Ticker: {data['ticker']}\n"
                f"Beta+: {data['beta_plus']}\n"
                f"Beta-: {data['beta_minus']}\n"
                f"Best Beta: {data['best_beta']}\n"
                f"Asymmetry Ratio: {data['asymmetry_ratio']}\n"
                f"CAPM Beta: {data['capm_beta']}\n"
            )
            self.result_box.setText(formatted)
        except requests.exceptions.RequestException as e:
            self.result_box.setText(f"Error fetching data:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DAPMWindow()
    window.show()
    sys.exit(app.exec())