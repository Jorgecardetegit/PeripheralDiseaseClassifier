document.addEventListener('DOMContentLoaded', function () {
    var dropArea = document.getElementById('drop-area');

    // Object to keep track of image data
    var imageData = {};

    // Add event listeners to checkboxes
    document.getElementById('classifications').addEventListener('change', updateDisplayedImage);
    document.getElementById('locations').addEventListener('change', updateDisplayedImage);
    document.getElementById('mask').addEventListener('change', updateDisplayedImage);

    // Function to create an image element
    function createImageElement(src) {
        var img = new Image();
        img.src = `data:image/jpeg;base64,${src}`;
        img.classList.add('uploaded-image'); // Add additional classes as necessary
        img.classList.add('uploaded-cell-image'); 
        return img;
    }

    // Function to update displayed image based on checkbox
    function updateDisplayedImage() {
        var processedImageContainer = document.getElementById('processed-image-container');
        processedImageContainer.innerHTML = ''; // Clear the previous image
    
        var imgElement;
    
        // Determine which image to show based on checkboxes
        var classificationsChecked = document.getElementById('classifications').checked;
        var locationsChecked = document.getElementById('locations').checked;
        var maskChecked = document.getElementById('mask').checked;
    
        // Check combinations and assign the appropriate image element
        if (maskChecked) {
            imgElement = createImageElement(imageData.mask);
        } else if (classificationsChecked && locationsChecked) {
            imgElement = createImageElement(imageData.processed_image);
        } else if (classificationsChecked) {
            imgElement = createImageElement(imageData.classification_image);
        } else if (locationsChecked) {
            imgElement = createImageElement(imageData.square_image);
        } else {
            // If no checkboxes are checked, show the original image
            imgElement = createImageElement(imageData.original_image);
        }
    
        // Append the chosen image element to the container
        processedImageContainer.appendChild(imgElement);
        displayExtractedCells();
        updateCellsDisplay(data.extracted_cells);
    }    

    function displayExtractedCells() {
        var cellsContainer = document.getElementById('cells-display');
        cellsContainer.innerHTML = '';  // Clear existing cell images
        imageData.extracted_cells.forEach(cellImage => {
            cellsContainer.appendChild(createImageElement(cellImage));
        });
    }

    function updateCellsDisplay(imagesEncoded) {
        var cellsDisplay = document.getElementById('cells-display');
        cellsDisplay.innerHTML = ''; // Clear the previous cell images
    
        imagesEncoded.forEach(imgSrc => {
            let imgElement = createCellImageElement(imgSrc);
            cellsDisplay.appendChild(imgElement); // Append each new cell image
        });
    }



    function updateTableWithData(cellsData) {
        var tableBody = document.getElementById('cells-info-body');
        tableBody.innerHTML = ''; // Clear existing rows
    
        cellsData.forEach(function(cell) {
            var row = `<tr>
                          <td>${cell.numberDetected}</td>
                          <td>${cell.disease}</td>
                          <td>${cell.confidence}</td>
                       </tr>`;
            tableBody.innerHTML += row; // Append the new row to the table body
        });
    }
    
    // Sample data
    var cellsData = [
        { numberDetected: 3, disease: 'CLL', confidence: '90%' },
        // ... more data
    ];
    
    // Call the function to update the table with the sample data

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Handle file selection via input
    var fileInput = document.getElementById('fileElem');
    fileInput.addEventListener('change', function (event) {
        handleFiles(this.files);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        var dt = e.dataTransfer;
        var files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        ([...files]).forEach(uploadFile);
    }

    function uploadFile(file) {
        var formData = new FormData();
        formData.append('image', file);
        fetch('/', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                imageData.original_image = data.original_image;
                imageData.processed_image = data.processed_image;
                imageData.mask = data.mask;
                imageData.square_image = data.square_image;
                imageData.classification_image = data.classification_image;
                imageData.extracted_cells = data.extracted_cells; // Store extracted cells
                updateDisplayedImage();  // Display the appropriate image
                updateCellsDisplay(data.extracted_cells);
                updateTableWithData(data.extracted_cells);
            })
            .catch(error => console.error('Fetch error:', error));
    }
    // This function receives the array of cell data and populates the table




function updateCellsInfoTable(cellsData) {
    const tableBody = document.getElementById('cells-info-body');
    tableBody.innerHTML = ''; // Clear existing table rows
  
    // Set the number of detected cells
    const cellsDetectedRow = document.createElement('tr');
    cellsDetectedRow.innerHTML = `<td colspan="3">${cellsData.length} cells detected</td>`;
    tableBody.appendChild(cellsDetectedRow);
  
    // Add a row for each cell
    cellsData.forEach((cell, index) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>Cell ${index + 1}</td>
        <td>${cell.disease}</td>
        <td>${cell.confidence}%</td>
      `;
      tableBody.appendChild(row);
    });
  }
  
  // Call this function with the array of cell data after fetching it from the server
  // updateCellsInfoTable(cellsDataFromServer);
  
});

