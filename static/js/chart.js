function buildChart(fake, real){

const ctx = document.getElementById("newsChart");

if(!ctx) return;

new Chart(ctx,{
type:'doughnut',
data:{
labels:['Fake News','Real News'],
datasets:[{
data:[fake,real]
}]
}
});

}