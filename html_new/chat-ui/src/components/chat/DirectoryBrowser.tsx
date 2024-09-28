import React, { useState, useEffect } from 'react';
import './DirectoryBrowser.css'; // Import CSS for styling
import { FolderSearch } from "lucide-react";

const servicesConfig = (window as any).ServiceConfig;

const DirectoryBrowser: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [currentPath, setCurrentPath] = useState<string>(''); // Root directory
  const [directories, setDirectories] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]); // To handle navigation history
  const [message, setMessage] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false); // Loading state
  const [error, setError] = useState<string | null>(null); // Error state

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
    setMessage('');
    setError(null);
  };

  // Fetch directories whenever currentPath changes
  useEffect(() => {
    const fetchDirectories = async () => {
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
        // const response = await fetch(`/api/directories?path=${encodeURIComponent(currentPath)}`);
        if (response.ok) {
          const data = await response.json();
          setDirectories(data.directories); // Assuming API returns { directories: [...] }
        } else {
          // Attempt to parse error message from response
          const errorData = await response.json();
          throw new Error(errorData.message || 'Failed to fetch directories.');
        }
      } catch (err: unknown) { // Explicitly typing err as unknown
        console.error('Error fetching directories:', err);
        if (err instanceof Error) { // Type guard to ensure err has a message property
          setError(err.message);
        } else {
          setError('An unknown error occurred while fetching directories.');
        }
        setDirectories([]);
      } finally {
        setLoading(false);
      }
    };

    fetchDirectories();
  }, [currentPath]);

  // Handle directory click
  const handleDirectoryClick = (dir: string) => {
    const newPath = currentPath ? `${currentPath}/${dir}` : dir;
    setHistory([...history, currentPath]);
    setCurrentPath(newPath);
  };

  // Handle navigating back
  const handleBack = () => {
    const previousPath = history.pop();
    setHistory([...history]);
    setCurrentPath(previousPath || '');
  };

  // Handle directory selection (final selection)
  const handleSelect = async () => {
    if (!currentPath) {
      setError('Please select a directory before proceeding.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(servicesConfig.envAPIEndpoint + '/entrypoint/set-searchsubdir', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ searchsubdir: currentPath }),
      });
      /*
      const response = await fetch('/api/select-directory', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ directory: currentPath }),
      });
      */

      if (response.ok) {
        const data = await response.json();
        setMessage(data.message || 'Directory selected successfully!');
        closeModal();
      } else {
        // Attempt to parse error message from response
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to select the directory.');
      }
    } catch (err: unknown) { // Explicitly typing err as unknown
      console.error('Error selecting directory:', err);
      if (err instanceof Error) { // Type guard to ensure err has a message property
        setError(err.message);
      } else {
        setError('An unknown error occurred while selecting the directory.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <FolderSearch size={16} className="folder-search-icon" opacity={ loading ? 0.5 : 1} onClick={openModal}/>

      {message && <p className="success-message">{message}</p>}

      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Browse Directories</h2>
            <div className="modal-content">
              <div className="navigation">
                <button onClick={handleBack} disabled={history.length === 0}>
                  Back
                </button>
                <span>Current Path: /{currentPath}</span>
              </div>

              {loading ? (
                <p>Loading directories...</p>
              ) : error ? (
                <p className="error-message">{error}</p>
              ) : (
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
              )}
            </div>
            <div className="modal-actions">
              <button onClick={closeModal} disabled={loading}>
                Cancel
              </button>
              <button onClick={handleSelect} disabled={!currentPath || loading}>
                {loading ? 'Selecting...' : 'Select This Directory'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DirectoryBrowser;
