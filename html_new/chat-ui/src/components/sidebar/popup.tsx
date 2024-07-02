import React from "react"

const { useState, useEffect, useRef } = React

const PopupMenu = () => {
  const [isShown, setIsShown] = useState(false)
  const popupRef = useRef()
  const documentClickHandler = useRef()
  
  useEffect(() => {
    // documentClickHandler.current  = (e:any) => {
    //   console.log('documentClickHandler')
      
    //   if (popupRef.current.contains(e.target)) return

    //   setIsShown(false)
    //   removeDocumentClickHandler()
    // }
  }, [])
  
  const removeDocumentClickHandler = () => {
    console.log('removeDocumentClickHandler')
    
    //document.removeEventListener('click', documentClickHandler.current)
  }
  
  const handleToggleButtonClick = () => {
    console.log('handleToggleButtonClick')
    
    if (isShown) return
    
    setIsShown(true)
    //document.addEventListener('click', documentClickHandler.current)
  }
  
  const handleCloseButtonClick = () => {
    console.log('handleCloseButtonClick')
    
    setIsShown(false)
    removeDocumentClickHandler()
  }
  
  return (
    <div className="popup-menu-container">
      <button onClick={handleToggleButtonClick}>
        Toggle Menu
      </button>
      <div
        className={`popup-menu ${isShown ? 'shown' : ''}`}
        //ref={popupRef}
      >
        <div>menu</div>
        <button onClick={handleCloseButtonClick}>
          Close Menu
        </button>
      </div>
    </div> 
  )
}


export default PopupMenu;