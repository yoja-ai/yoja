import React, { useEffect, useState } from 'react';
import './IndexingProgress.css'; // Import the CSS file

const servicesConfig = (window as any).ServiceConfig;

const ProgressBar = () => {
  const [progress, setProgress] = useState(0); // initial progress value
  const [loading, setLoading] = useState(true); // loading state
  const [isHovered, setIsHovered] = useState(false); // state to track hover

  useEffect(() => {
    // Function to fetch progress data from API
    const fetchProgress = async () => {
      try {
        const requestUrl = servicesConfig.envAPIEndpoint + "/entrypoint/get-indexing-progress";
        const response = await fetch(requestUrl);
        const data = await response.json();
        var finished = 0;
        var not_finished = 0;
        if (data.hasOwnProperty('gdrive_unmodified_size')) {
          finished += data.gdrive_unmodified_size;
        }
        if (data.hasOwnProperty('gdrive_needs_embedding_size')) {
          not_finished += data.gdrive_needs_embedding_size;
        }
        if (data.hasOwnProperty('dropbox_unmodified_size')) {
          finished += data.dropbox_unmodified_size;
        }
        if (data.hasOwnProperty('dropbox_needs_embedding_size')) {
          not_finished += data.dropbox_needs_embedding_size;
        }
        var pct;
        if (finished > 0 || not_finished > 0) {
          pct = (finished/(finished+not_finished))*100;
        } else {
          pct = 0;
        }
        setProgress(pct);
      } catch (error) {
        console.error('Error fetching progress:', error);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch when the component mounts
    fetchProgress();

    // Set an interval to fetch progress every 60 seconds (60000 ms)
    const interval = setInterval(fetchProgress, 60000);

    // Cleanup the interval on component unmount
    return () => clearInterval(interval);
  }, []); // empty dependency array ensures the effect runs only once when the component mounts

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div
      className="wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="progress-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      {isHovered && <p className="progress-text">Indexing: {progress}% Complete</p>}
    </div>
  );
};

export default ProgressBar;
