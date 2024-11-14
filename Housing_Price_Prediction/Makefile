# Makefile for running ETL and Machine Learning pipeline

# Define directories
CONFIG_DIR = Config_Files
ETL_DIR = Housing_Data_Processing
ML_DIR = Model_Training
PROCESSED_DATA_DIR = processed_data
PARSER_DIR = Parser

# Targets
.PHONY: all etl ml clean

# Ensure CONFIG_FILE is provided for etl and ml targets
ifeq ($(CONFIG_FILE),)
$(error CONFIG_FILE is not defined. Please provide it, e.g., 'make all CONFIG_FILE=your_config.yaml')
endif

# Run the full ETL and ML pipeline
all: etl ml
	@echo "ETL and ML pipeline completed successfully."

# Run the ETL process
etl:
	@echo "Running ETL pipeline..."
	PYTHONPATH=$(PARSER_DIR):. python3 $(ETL_DIR)/etl_main.py $(CONFIG_DIR)/$(CONFIG_FILE)
	@echo "ETL pipeline completed."

# Run the ML training process
ml:
	@echo "Running ML training pipeline..."
	PYTHONPATH=$(PARSER_DIR):. python3 $(ML_DIR)/main.py $(CONFIG_DIR)/$(CONFIG_FILE)
	@echo "ML training completed."

# Clean up generated model and result files
clean:
	@echo "Cleaning up saved models and results..."
	rm -rf $(ML_DIR)/saved_model_params_cvres/*
	rm -rf $(PROCESSED_DATA_DIR)/*
	@echo "Cleanup completed."

