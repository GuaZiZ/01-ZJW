// pages/func/func.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    banners: [{
               'src': 'images/2.png'
           },
           {
               'src': 'images/3.png'
           },
           {
               'src': 'images/4.png'
           },
           {
               'src': 'images/5.png'
           },
           {
               'src': 'images/6.png'
           },
           {
               'src': 'images/7.png'
           },
           {
               'src': 'images/8.png'
           },
           {
               'src': 'images/9.png'
           }],
     indicatorDots: true,
     vertical: false,
     autoplay: true,
     interval: 2000,
     duration: 500,
     circular: true
 },
 begin1() {
  wx.navigateTo({
    url: '/pages/jdc/jdc'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin2() {
  wx.navigateTo({
    url: '/pages/rt/rt'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin3() {
  wx.navigateTo({
    url: '/pages/lmkw/lmkw'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin4() {
  wx.navigateTo({
    url: '/pages/dljs/dljs'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin5() {
  wx.navigateTo({
    url: '/pages/ldlj/ldlj'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin6() {
  wx.navigateTo({
    url: '/pages/xt/xt'   //url后面是跳转地址（备注不要复制进去）
  })
},
 begin7() {
  wx.navigateTo({
    url: '/pages/fjdc/fjdc'   //url后面是跳转地址（备注不要复制进去）
  })
},

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})