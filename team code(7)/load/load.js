// pages/list/list.js
Page({
  /**
   * 页面的初始数据
   */
  data: {
    timer:"",
    deflautWidth:0,
  },
  
  begin() {
    wx.navigateTo({
      url: '/pages/load/load' //url后面是跳转地址（备注不要复制进去）
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  // home:function(){
  //   wx.switchTab({
  //     url: '/pages/main/main'
  //   })
  // },

  onLoad(options) {
    let index = 0;
    let that = this
    this.data.timer=setInterval(() => { //注意箭头函数！！
      index += 10;
      that.setData({
        deflautWidth: index
      })
      if (that.data.deflautWidth == 100) {
        clearInterval(this.data.timer);
        // wx.switchTab({
        //   url: '/pages/main/main'
        // })
      }
    }, 1000);
    setTimeout(function () {
      wx.reLaunch({
        url: '../main/main',
      })
    }, 10000);
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