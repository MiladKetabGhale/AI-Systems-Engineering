# Makefile for running ETL and Machine Learning pipeline

# Define directories
CONFIG_DIR=Config_Files
ETL_DIR=Housing_Data_Processing
ML_DIR=Model_Training
PROCESSED_DATA_DIR=processed_data

# Targets
.PHONY: all etl ml clean

# Check if CONFIG_FILE is provided (only for ETL and ML targets)
ifeq ($(filter etl ml all,$(MAKECMDGOALS)),)
CONFIG_FILE_DEFINED := true
else ifeq ($(CONFIG_FILE),)
$(error CONFIG_FILE is not defined. Please provide the config file name as an argument, e.g., 'make all CONFIG_FILE=your_config.cfg')
endif

# Run the full ETL and ML pipeline
all: etl ml
	@echo "ETL and ML pipeline completed successfully."

# Run the ETL process
etl:
	@echo "Running ETL pipeline..."
	python3 $(ETL_DIR)/etl_main.py $(CONFIG_DIR)/$(CONFIG_FILE)
	@echo "ETL pipeline completed."

# Run the ML training process
ml:
	@echo "Running ML training pipeline..."
	PYTHONPATH=. python3 $(ML_DIR)/main.py -c $(CONFIG_DIR)/$(CONFIG_FILE)
	@echo "ML training completed."

# Clean up generated model and result files
clean:
	@echo "Cleaning up saved models and results..."
	rm -rf $(ML_DIR)/saved_model_params_cvres/*
	rm -rf $(PROCESSED_DATA_DIR)/*
	@echo "Cleanup completed."

