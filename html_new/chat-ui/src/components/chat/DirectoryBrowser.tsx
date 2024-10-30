import React, { useState, useEffect } from 'react';
import { SquareChevronRight } from "lucide-react";
import './DirectoryBrowser.css';

const servicesConfig = (window as any).ServiceConfig;

// Define types for directory and file items
interface DirectoryData {
  directories: string[];
  files: string[];
}

interface DirectoryBrowserProps {
  onFileSelect: (filePath: string) => void; // Callback prop for selected file
}

const DirectoryBrowser: React.FC<DirectoryBrowserProps> = ({ onFileSelect }) => {
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [currentPath, setCurrentPath] = useState<string>(''); // Root directory
  const [directories, setDirectories] = useState<string[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]); // To handle navigation history
  const [loading, setLoading] = useState<boolean>(false); // Loading state
  const [error, setError] = useState<string | null>(null); // Error state
  const [selectedFile, setSelectedFile] = useState<string | null>(null); // Selected file

  // Function to open the modal and fetch root directories
  const openModal = () => {
    setIsModalOpen(true);
    setCurrentPath('');
    setHistory([]);
    setError(null);
  };

  // Function to close the modal
  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedFile(null); // Clear selected file when closing
    setError(null);
  };

  // Fetch directories and files whenever currentPath changes
  useEffect(() => {
    const fetchDirectoryData = async () => {
      setLoading(true);
      setError(null);
      try {
        let headers: any = { "Content-Type": "application/json" };
        const API_URL = "/entrypoint/directory-browser";
        const requestUrl = servicesConfig.envAPIEndpoint + API_URL;
        const requestBody = JSON.stringify({
          parentdir: currentPath
        });
        let response = await fetch(requestUrl, {
          method: "POST",
          headers,
          body: requestBody,
        });
        if (response.ok) {
          const data: DirectoryData = await response.json();
          setDirectories(data.directories); // Assuming API returns { directories: [...] }
          setFiles(data.files);             // Assuming API returns { files: [...] }
        } else {
          const errorData = await response.json();
          throw new Error(errorData.message || 'Failed to fetch directory data.');
        }
      } catch (err: unknown) {
        console.error('Error fetching directory data:', err);
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('An unknown error occurred while fetching directory data.');
        }
        setDirectories([]);
        setFiles([]);
      } finally {
        setLoading(false);
      }
    };

    fetchDirectoryData();
  }, [currentPath]);

  // Handle directory click
  const handleDirectoryClick = (dir: string) => {
    const newPath = currentPath ? `${currentPath}/${dir}` : dir;
    setHistory([...history, currentPath]);
    setCurrentPath(newPath);
  };

  // Handle file click
  const handleFileClick = (file: string) => {
    setSelectedFile(`${currentPath}/${file}`);
  };

  // Handle navigating back
  const handleBack = () => {
    const previousPath = history.pop();
    setHistory([...history]);
    setCurrentPath(previousPath || '');
    setSelectedFile(null); // Clear selected file when navigating back
  };

  // Handle file selection (final selection)
  const handleSelectFile = () => {
    if (!selectedFile) {
      setError('Please select a file before proceeding.');
      return;
    }

    // Pass the selected file path to the parent component
    onFileSelect(selectedFile);
    closeModal();
  };

  return (
    <div>
      <div className="sidebar-header-icon" onClick={openModal}>
        <SquareChevronRight size={14} strokeWidth={2} />
      </div>

      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Browse Directories and Files</h2>
            <div className="modal-content">
              <div className="navigation">
                <button onClick={handleBack} disabled={history.length === 0}>
                  Back
                </button>
                <span>Current Path: /{currentPath}</span>
              </div>

              {loading ? (
                <p>Loading directories and files...</p>
              ) : error ? (
                <p className="error-message">{error}</p>
              ) : (
                <>
                  <ul className="directory-list">
                    {directories.length > 0 ? (
                      directories.map((dir) => (
                        <li key={dir} onClick={() => handleDirectoryClick(dir)}>
                          📁 {dir}
                        </li>
                      ))
                    ) : (
                      <li>No directories found.</li>
                    )}
                  </ul>

                  <ul className="file-list">
                    {files.length > 0 ? (
                      files.map((file) => (
                        <li
                          key={file}
                          onClick={() => handleFileClick(file)}
                          className={selectedFile === `${currentPath}/${file}` ? 'selected' : ''}
                        >
                          📄 {file}
                        </li>
                      ))
                    ) : (
                      <li>No files found in this directory.</li>
                    )}
                  </ul>
                </>
              )}
            </div>
            <div className="modal-actions">
              <button onClick={closeModal}>
                Cancel
              </button>
              <button onClick={handleSelectFile} disabled={!selectedFile}>
                Select This File
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DirectoryBrowser;
