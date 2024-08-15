
import React, { Fragment } from "react"
import { AppstoreOutlined, MailOutlined, SettingOutlined } from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { Menu, Modal } from 'antd';
import SubMenu from "antd/es/menu/SubMenu";
import SettingsModel from "./SettingsModel";
import { Avatar, AvatarImage } from "../ui/avatar";
import { LogOut, Settings } from "lucide-react";

declare var window: any
const { useState, useEffect, useRef } = React

const UserMenu = ({isCollapsed}: any) => {
  const [isShown, setIsShown] = useState(false)
  const popupRef = useRef()
  const documentClickHandler = useRef()
  let accessToken: any = null;
  
  type MenuItem = Required<MenuProps>['items'][number];

  const items: MenuItem[] = [
    {
      key: 'sub1',
      icon: <MailOutlined />,
      label: 'Navigation One',
    },
    {
      key: 'sub2',
      icon: <AppstoreOutlined />,
      label: 'Navigation Two',
      children: [
        { key: '5', label: 'Option 5' },
        { key: '6', label: 'Option 6' },
        {
          key: 'sub3',
          label: 'Submenu',
          children: [
            { key: '7', label: 'Option 7' },
            { key: '8', label: 'Option 8' },
          ],
        },
      ],
    },
    {
      key: 'sub4',
      label: 'Navigation Three',
      icon: <SettingOutlined />,
      children: [
        { key: '9', label: 'Option 9' },
        { key: '10', label: 'Option 10' },
        { key: '11', label: 'Option 11' },
        { key: '12', label: 'Option 12' },
      ],
    },
  ];

  const onClick: MenuProps['onClick'] = (e) => {
    console.log('click', e);
  };

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

  const [isModalOpen, setIsModalOpen] = useState(false);

  const showModal = () => {
    setIsModalOpen(true);
  };

  const handleOk = () => {
    setIsModalOpen(false);
  };

  const handleCancel = () => {
    setIsModalOpen(false);
  };

  const handleSignoutClick = () => {
    if (accessToken) {
      accessToken = null;
      google.accounts.oauth2.revoke();
      window.location.href="login.html"
    }
    window.location.href="login.html"
  }
  
  if (!isCollapsed) {
    return (
      <>
        <div className="popup-menu-container">
          <Menu key="user">
            <SubMenu
              title={
                <Fragment>
                  <div className='sidebar-user-avatar'>
                  <Avatar className="flex justify-center items-center">
                    <AvatarImage
                      src={window.picture}
                      width={6}
                      height={6}
                    />
                  </Avatar>
                  </div>
                  <div className='sidebar-user-info'>
                    <span className='sidebar-username'> {window.fullname} </span>
                    <span className='sidebar-user-eamil'> {window.google} </span>
                  </div>
                </Fragment>
              }
            >
              <Menu.Item key="Settings" onClick={showModal}>
                <span style={{display: 'flex', justifyContent:'center', alignItems:'center', gap: '5px'}}> <Settings size={16} /> Settings </span>
              </Menu.Item>
              <Menu.Item key="SignOut"  onClick={handleSignoutClick}>
                <div style={{display: 'flex', justifyContent:'center', alignItems:'center',  gap: '5px'}}> <LogOut size={16} /> Sign Out </div>
              </Menu.Item>
            </SubMenu>
          </Menu>
        </div> 
        <Modal title="Settings" open={isModalOpen} onOk={handleOk} onCancel={handleCancel} footer={null} width={660} height={600}>
          <SettingsModel />
        </Modal>
      </>
    )
  } else {
    return null
  }
}


export default UserMenu;

